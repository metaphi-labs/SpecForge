# coding=utf-8
"""Online training wrapper for single-layer MTP (architecture-independent).

MTP predicts the next token from the current token's embedding plus the target
model's last hidden state.  Shift is performed inside this wrapper; the target
backend is expected to return *raw* input_ids and last_hidden_states (DFlash
style), not the pre-shifted output of generate_eagle3_data.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_step_weights(beta: float = 0.6, num_steps: int = 3) -> List[float]:
    """Compute normalized exponential-decay step weights.

    alpha_k = beta^(k-1) / sum(beta^(j-1) for j=1..K)

    See FastMTP (arXiv:2509.18362), Equation 2.
    """
    raw = [beta**k for k in range(num_steps)]
    total = sum(raw)
    return [w / total for w in raw]


class OnlineMTPModel(nn.Module):
    """
    Online MTP training wrapper.

    Architecture-agnostic: any registered MTP draft module exposing
    ``forward_hidden(input_ids, hidden_states, attention_mask, position_ids)``,
    a position-wise ``mtp.lm_head``, and a ``config`` with ``pad_token_id`` can
    be plugged in (see
    ``specforge/modeling/draft/mtp/``).

    Args:
        draft_model: The MTP draft model (e.g. ``modeling/draft/mtp/qwen3_5.py``).
        ploss_decay: Per-layer loss decay.  For a single MTP layer this is
            unused, but kept for multi-layer extension.
        objective_chunk_size: Token positions per lm_head+CE chunk.  Bounds the
            full-vocab logits to one chunk at a time (with activation
            checkpointing), instead of materializing [batch*seq, vocab] twice.
            0 disables chunking. Configured by ``training.mtp_objective_chunk_size``
            in YAML. This bounds objective intermediates only; backbone
            activations, attention, weights, and optimizer state are unchanged.
        num_speculative_steps: Number of teacher-forced draft steps per
            position (1 = single-step native fine-tune, the default).
        step_weight_beta: FastMTP exponential-decay base for per-step loss
            weights (only used when num_speculative_steps > 1).
        step_weights: Explicit per-step loss weights; overrides
            ``step_weight_beta`` when given.
        chain_context: What a draft step k >= 2 attends to. ``"sequence"``
            (FastMTP / #786): every step is a full causal pass, so position t
            sees the step-k entries of earlier positions. ``"prefix"``: the
            recursion of SGLang's EAGLE/NEXTN worker -- position t sees the
            step-1 entries of the prefix (what draft-extend wrote into the
            draft KV cache) plus only its own chain's entries.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        ploss_decay: float = 1.0,
        objective_chunk_size: int = 4096,
        num_speculative_steps: int = 1,
        step_weight_beta: float = 0.6,
        step_weights: Optional[List[float]] = None,
        chain_context: str = "sequence",
    ) -> None:
        super().__init__()
        if objective_chunk_size < 0:
            raise ValueError(
                f"objective_chunk_size must be >= 0, got {objective_chunk_size}"
            )
        if num_speculative_steps < 1:
            raise ValueError(
                f"num_speculative_steps must be >= 1, got {num_speculative_steps}"
            )
        self.draft_model = draft_model
        self.ploss_decay = ploss_decay
        self.objective_chunk_size = objective_chunk_size
        self.num_speculative_steps = num_speculative_steps
        self.step_weight_beta = step_weight_beta
        if step_weights is None and num_speculative_steps > 1:
            step_weights = compute_step_weights(step_weight_beta, num_speculative_steps)
        if step_weights is not None and len(step_weights) != num_speculative_steps:
            raise ValueError(
                f"step_weights has {len(step_weights)} entries but "
                f"num_speculative_steps={num_speculative_steps}"
            )
        self.step_weights = step_weights
        if chain_context not in ("sequence", "prefix"):
            raise ValueError(
                f"chain_context must be 'sequence' or 'prefix', got {chain_context!r}"
            )
        self.chain_context = chain_context

    def _shift_for_next_token(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        offset: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shift labels/mask to match vLLM speculative decoding.

        In serving, the draft model's input_ids are the target input_ids shifted
        right by one (the draft fuses token x_{t+1} with target hidden state
        h_t) and it predicts the token after that (x_{t+2}). Training therefore
        uses:
          - draft input: input_ids[:, 1:]  (x_1..x_T, padded)
          - label:       x_2..x_T followed by a pad (length matches logits)
        """
        # x_2..x_T has length seq_len-2; pad one position so its length equals
        # seq_len-1 (same as the shifted hidden states). The padded position is
        # ignored.
        # Draft step k (0-based) predicts x_{t+k+2}: offset = k + 2.
        shift_labels = F.pad(input_ids[:, offset:], (0, offset - 1), value=-100)
        shift_mask = F.pad(loss_mask[:, offset:], (0, offset - 1), value=0)
        return shift_labels, shift_mask

    def _chunked_objective(
        self,
        shift_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        offset: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply lm_head + cross-entropy in bounded chunks over positions.

        ``shift_hidden`` drops the final position up front (lm_head is
        position-wise, so this equals shifting the logits), and each chunk's
        logits are freed (or recomputed under checkpointing) before the next
        chunk runs.
        """
        shift_labels, shift_mask = self._shift_for_next_token(
            input_ids, loss_mask, offset
        )
        batch, positions, hidden_size = shift_hidden.shape
        flat_hidden = shift_hidden.reshape(batch * positions, hidden_size)
        flat_labels = shift_labels.reshape(-1)
        flat_mask = shift_mask.reshape(-1)
        rows = flat_hidden.shape[0]
        chunk_size = self.objective_chunk_size if self.objective_chunk_size else rows
        lm_head = self.draft_model.mtp.lm_head

        def _chunk_terms(hidden_chunk, labels_chunk, mask_chunk):
            logits = lm_head(hidden_chunk)
            losses = F.cross_entropy(logits, labels_chunk, reduction="none")
            with torch.no_grad():
                corrects = (logits.argmax(dim=-1) == labels_chunk).float()
                corrects = corrects * mask_chunk
            return (losses * mask_chunk.float()).sum(), corrects, mask_chunk.sum()

        loss_num = None
        denom = None
        corrects_chunks = []
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            chunk_args = (
                flat_hidden[start:end],
                flat_labels[start:end],
                flat_mask[start:end],
            )
            if (
                chunk_size < rows
                and torch.is_grad_enabled()
                and flat_hidden.requires_grad
            ):
                chunk_terms = torch.utils.checkpoint.checkpoint(
                    _chunk_terms, *chunk_args, use_reentrant=False
                )
            else:
                chunk_terms = _chunk_terms(*chunk_args)
            chunk_loss_num, chunk_corrects, chunk_denom = chunk_terms
            loss_num = chunk_loss_num if loss_num is None else loss_num + chunk_loss_num
            denom = chunk_denom if denom is None else denom + chunk_denom
            corrects_chunks.append(chunk_corrects)

        loss = loss_num / denom.clamp_min(1)
        corrects = torch.cat(corrects_chunks).view(batch, positions)
        return loss, corrects, shift_mask.float()

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            input_ids: raw token ids, [batch, seq_len].
            hidden_states: target model last hidden states, [batch, seq_len, hidden].
            loss_mask: [batch, seq_len].
            attention_mask: optional padding mask, [batch, seq_len].
            position_ids: optional position ids, [batch, seq_len].

        Returns:
            loss: scalar weighted loss.
            acc_corrects: per-layer per-position correct tensors.
            acc_denoms: per-layer per-position denominator tensors.
        """
        if self.num_speculative_steps > 1:
            return self._forward_multi_step(
                input_ids, hidden_states, loss_mask, attention_mask, position_ids
            )

        # Draft input is the target sequence shifted right by one.  The last
        # position is padded because there is no x_{T+1}; its hidden state is
        # dropped before the objective in _chunked_objective.
        pad_token_id = getattr(self.draft_model.config, "pad_token_id", 0)
        shifted_input_ids = F.pad(input_ids[:, 1:], (0, 1), value=pad_token_id)

        # The padding mask must follow the same shift so the synthetic pad token
        # at the last position is not attended to.
        if attention_mask is not None:
            shifted_attention_mask = F.pad(attention_mask[:, 1:], (0, 1), value=0).to(
                attention_mask.dtype
            )
        else:
            shifted_attention_mask = None

        # Serving evaluates the shifted draft token x[t+1] at its own position
        # p[t+1], even though it is fused with the target hidden state h[t].
        # Preserve caller-supplied offsets (for packed/non-zero-based sequences)
        # and give the synthetic final token the next monotonic position.
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        elif position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must have the same [batch, seq_len] shape as "
                f"input_ids; got {tuple(position_ids.shape)} and "
                f"{tuple(input_ids.shape)}"
            )
        shifted_position_ids = torch.cat(
            (position_ids[:, 1:], position_ids[:, -1:] + 1), dim=1
        )

        draft_hidden = self.draft_model.forward_hidden(
            input_ids=shifted_input_ids,
            hidden_states=hidden_states,
            attention_mask=shifted_attention_mask,
            position_ids=shifted_position_ids,
        )

        # The synthetic pad position has no label; drop it before the objective.
        loss, corrects, denoms = self._chunked_objective(
            draft_hidden[:, :-1], input_ids, loss_mask
        )

        # Single-layer MTP: wrap in length-1 lists for E1 evaluator compatibility.
        return loss, [corrects], [denoms]

    def _forward_multi_step(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Teacher-forced multi-step MTP training (FastMTP-style).

        At step k (0-based) the draft consumes the ground-truth token
        x[t+k+1] (embedded) at its own position p[t+k+1], fused with the
        previous step's MTP output at index t (the target's last hidden state
        at step 0), and predicts x[t+k+2]. This is how a single MTP head is
        reused recursively at serving time. Each step's loss is weighted by
        ``step_weights`` (normalized exponential decay, FastMTP Eq. 2), and
        every step uses the same chunked lm_head + CE objective as the
        single-step path. Step 0 is exactly the single-step objective.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        pad_token_id = getattr(self.draft_model.config, "pad_token_id", 0)
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        elif position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must have the same [batch, seq_len] shape as "
                f"input_ids; got {tuple(position_ids.shape)} and "
                f"{tuple(input_ids.shape)}"
            )

        steps = min(self.num_speculative_steps, max(1, seq_len - 2))
        total_loss = None
        corrects_per_step: List[torch.Tensor] = []
        denoms_per_step: List[torch.Tensor] = []
        current_hidden = hidden_states
        chain_cache = {} if self.chain_context == "prefix" else None
        for step in range(steps):
            shift = step + 1
            step_ids = F.pad(input_ids[:, shift:], (0, shift), value=pad_token_id)
            step_attention = (
                F.pad(attention_mask[:, shift:], (0, shift), value=0).to(
                    attention_mask.dtype
                )
                if attention_mask is not None
                else None
            )
            # Positions of the shifted tokens, extended monotonically past the end.
            tail = position_ids[:, -1:] + torch.arange(
                1, shift + 1, device=device, dtype=position_ids.dtype
            ).unsqueeze(0)
            step_positions = torch.cat((position_ids[:, shift:], tail), dim=1)

            if chain_cache is None:
                draft_hidden = self.draft_model.forward_hidden(
                    input_ids=step_ids,
                    hidden_states=current_hidden,
                    attention_mask=step_attention,
                    position_ids=step_positions,
                )
            else:
                # Step 0 records the prefix K/V; later steps are chain steps.
                draft_hidden = self.draft_model.forward_hidden(
                    input_ids=step_ids,
                    hidden_states=current_hidden,
                    attention_mask=step_attention if step == 0 else None,
                    position_ids=step_positions,
                    chain_cache=chain_cache,
                )
            loss, corrects, denoms = self._chunked_objective(
                draft_hidden[:, :-1], input_ids, loss_mask, offset=step + 2
            )
            weighted = self.step_weights[step] * loss
            total_loss = weighted if total_loss is None else total_loss + weighted
            corrects_per_step.append(corrects)
            denoms_per_step.append(denoms)
            # The next step fuses its token embeddings with this step's output.
            current_hidden = draft_hidden

        return total_loss, corrects_per_step, denoms_per_step
