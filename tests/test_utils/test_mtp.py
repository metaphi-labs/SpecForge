# coding=utf-8
"""CPU unit tests for the MTP (native-head fine-tune) algorithm.

Covers the pieces that do not need a GPU or a real target checkpoint:
  - OnlineMTPModel forward/shift/loss/accuracy plumbing
  - MTPTrainStrategy batch -> StepOutput adaptation
  - strict native ``mtp.*`` weight initialization from a target checkpoint
  - selective checkpoint loading (modeling/target/checkpoint.py)
  - merge-back round trip (export/mtp.py)
  - draft-architecture and built-in algorithm registration
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.mtp.providers import _init_from_native_mtp
from specforge.core.mtp import OnlineMTPModel, compute_step_weights
from specforge.modeling.draft import available_drafts, resolve_draft
from specforge.modeling.draft.mtp import Qwen3_5MTPDraftModel
from specforge.training.strategies.base import MTPTrainStrategy


def _tiny_config(**overrides) -> Qwen3Config:
    payload = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rms_norm_eps=1e-6,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        mtp_config={"share_lm_head": True},
        tie_word_embeddings=False,
    )
    payload.update(overrides)
    config = Qwen3Config(**payload)
    config._attn_implementation = "eager"
    return config


def _tiny_batch(config: Qwen3Config, seq_len: int = 16, batch: int = 2):
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    loss_mask = torch.ones(batch, seq_len)
    return input_ids, hidden_states, loss_mask


def _cfg(target_model_path: str, draft_checkpoint_path: str = ""):
    return SimpleNamespace(
        model=SimpleNamespace(
            target_model_path=target_model_path,
            draft_checkpoint_path=draft_checkpoint_path,
            cache_dir=None,
        )
    )


class OnlineMTPModelTest(unittest.TestCase):
    def test_forward_returns_finite_loss_and_accuracy_lists(self):
        config = _tiny_config()
        model = OnlineMTPModel(Qwen3_5MTPDraftModel(config))
        input_ids, hidden_states, loss_mask = _tiny_batch(config)

        loss, corrects, denoms = model(
            input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask
        )

        self.assertEqual(0, loss.dim())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(1, len(corrects))
        self.assertEqual(1, len(denoms))
        # next-token shift drops one position
        self.assertEqual((2, 15), corrects[0].shape)
        self.assertEqual((2, 15), denoms[0].shape)

    def test_loss_backward_populates_mtp_grads(self):
        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        model = OnlineMTPModel(draft)
        input_ids, hidden_states, loss_mask = _tiny_batch(config)

        loss, _, _ = model(
            input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask
        )
        loss.backward()

        self.assertIsNotNone(draft.mtp.fc.weight.grad)
        self.assertTrue(torch.isfinite(draft.mtp.fc.weight.grad).all())

    def test_chunked_objective_matches_unchunked(self):
        self._assert_objective_parity(torch.device("cpu"), torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_chunked_objective_cuda_matches_legacy(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._assert_objective_parity(torch.device("cuda"), dtype)

    def _assert_objective_parity(self, device, dtype):
        config = _tiny_config()
        torch.manual_seed(0)
        input_ids, hidden_states, loss_mask = _tiny_batch(config)
        # ragged supervision so chunk boundaries cross masked/unmasked rows
        loss_mask[0, 5:9] = 0.0
        input_ids, loss_mask = input_ids.to(device), loss_mask.to(device)
        hidden_states = hidden_states.to(device=device, dtype=dtype)
        for frozen_head in (False, True):
            draft = Qwen3_5MTPDraftModel(config).to(device=device, dtype=dtype)
            draft.mtp.lm_head.requires_grad_(not frozen_head)
            draft.embed_tokens.requires_grad_(not frozen_head)
            results = {}
            for chunk_size in ("legacy", 0, 4, 4096):
                current = copy.deepcopy(draft)
                inputs = hidden_states.detach().clone().requires_grad_()
                if chunk_size == "legacy":
                    # Independent pre-chunking objective, including the full
                    # logits projection and contiguous shifted-logits copy.
                    shifted_ids = torch.nn.functional.pad(input_ids[:, 1:], (0, 1))
                    positions = (
                        torch.arange(1, input_ids.shape[1] + 1, device=device)
                        .unsqueeze(0)
                        .expand_as(input_ids)
                    )
                    logits = (
                        current(
                            input_ids=shifted_ids,
                            hidden_states=inputs,
                            position_ids=positions,
                        )
                        .logits[:, :-1]
                        .contiguous()
                    )
                    labels = torch.nn.functional.pad(
                        input_ids[:, 2:], (0, 1), value=-100
                    )
                    mask = torch.nn.functional.pad(loss_mask[:, 2:], (0, 1))
                    losses = torch.nn.functional.cross_entropy(
                        logits.view(-1, config.vocab_size),
                        labels.reshape(-1),
                        reduction="none",
                    )
                    loss = (losses * mask.reshape(-1)).sum() / mask.sum().clamp_min(1)
                    corrects = [(logits.argmax(-1) == labels).float() * mask]
                    denoms = [mask]
                else:
                    model = OnlineMTPModel(current, objective_chunk_size=chunk_size)
                    loss, corrects, denoms = model(
                        input_ids=input_ids, hidden_states=inputs, loss_mask=loss_mask
                    )
                loss.backward()
                gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in current.named_parameters()
                    if parameter.requires_grad
                }
                gradients["input_hidden_states"] = inputs.grad.detach().clone()
                results[chunk_size] = (loss.detach(), corrects[0], denoms[0], gradients)

            reference = results["legacy"]
            tolerances = (
                dict(rtol=2e-2, atol=2e-3)
                if dtype == torch.bfloat16
                else dict(rtol=1e-5, atol=1e-6)
            )
            for chunk_size in (0, 4, 4096):
                with self.subTest(frozen_head=frozen_head, chunk_size=chunk_size):
                    actual = results[chunk_size]
                    torch.testing.assert_close(actual[0], reference[0], **tolerances)
                    self.assertTrue(torch.equal(actual[1], reference[1]))
                    self.assertTrue(torch.equal(actual[2], reference[2]))
                    self.assertEqual(actual[3].keys(), reference[3].keys())
                    for name in reference[3]:
                        with self.subTest(parameter=name):
                            torch.testing.assert_close(
                                actual[3][name], reference[3][name], **tolerances
                            )

    def test_objective_chunk_size_rejects_negative(self):
        with self.assertRaisesRegex(ValueError, "objective_chunk_size"):
            OnlineMTPModel(
                Qwen3_5MTPDraftModel(_tiny_config()), objective_chunk_size=-1
            )

    def test_shift_for_next_token_matches_serving_alignment(self):
        model = OnlineMTPModel(Qwen3_5MTPDraftModel(_tiny_config()))
        input_ids = torch.tensor([[10, 11, 12, 13, 14]])
        loss_mask = torch.ones(1, 5)

        shift_labels, shift_mask = model._shift_for_next_token(input_ids, loss_mask)

        # labels are x_2..x_T padded with one ignore index
        self.assertEqual([12, 13, 14, -100], shift_labels[0].tolist())
        # the padded position is masked out
        self.assertEqual([1, 1, 1, 0], shift_mask[0].tolist())

    def test_forward_shifts_position_ids_with_draft_tokens(self):
        class _RecordingDraft(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(pad_token_id=0)
                self.position_ids = None
                self.mtp = SimpleNamespace(lm_head=torch.nn.Linear(8, 32, bias=False))

            def forward_hidden(
                self,
                input_ids,
                hidden_states,
                attention_mask=None,
                position_ids=None,
            ):
                self.position_ids = position_ids.detach().clone()
                return torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    8,
                    requires_grad=True,
                )

        draft = _RecordingDraft()
        model = OnlineMTPModel(draft)
        model(
            input_ids=torch.tensor([[10, 11, 12, 13]]),
            hidden_states=torch.zeros(1, 4, 8),
            loss_mask=torch.ones(1, 4),
            position_ids=torch.tensor([[4, 5, 6, 7]]),
        )

        # x[t+1] is fused with h[t], but RoPE must use x[t+1]'s serving
        # position. The synthetic final token is assigned the next position.
        self.assertEqual([[5, 6, 7, 8]], draft.position_ids.tolist())


def _single_step_reference(draft, input_ids, hidden_states, loss_mask):
    """The single-step objective spelled out independently of OnlineMTPModel:
    x[t+1] at position p[t+1] fused with h[t] predicts x[t+2]."""
    batch, seq_len = input_ids.shape
    pad = draft.config.pad_token_id
    ids = torch.nn.functional.pad(input_ids[:, 1:], (0, 1), value=pad)
    positions = torch.arange(1, seq_len + 1).unsqueeze(0).expand(batch, -1)
    out = draft.forward_hidden(ids, hidden_states, position_ids=positions)[:, :-1]
    logits = draft.mtp.lm_head(out)
    labels = input_ids[:, 2:]
    mask = loss_mask[:, 2:].float()
    ce = torch.nn.functional.cross_entropy(
        logits[:, : seq_len - 2].reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    )
    return (ce * mask.reshape(-1)).sum() / mask.sum()


class MultiStepMTPTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = _tiny_config()
        self.draft = Qwen3_5MTPDraftModel(self.config)
        self.batch = _tiny_batch(self.config, seq_len=24)

    def test_single_step_loss_unchanged(self):
        input_ids, hidden_states, loss_mask = self.batch
        loss, corrects, denoms = OnlineMTPModel(self.draft)(
            input_ids, hidden_states, loss_mask
        )
        ref = _single_step_reference(self.draft, input_ids, hidden_states, loss_mask)
        torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-6)
        self.assertEqual(1, len(corrects))

    def test_step_weights_normalized_exponential_decay(self):
        w = compute_step_weights(0.6, 3)
        self.assertAlmostEqual(1.0, sum(w))
        self.assertAlmostEqual(0.6, w[1] / w[0])
        self.assertAlmostEqual(0.6, w[2] / w[1])

    def test_k_step_with_only_first_weight_equals_single_step(self):
        input_ids, hidden_states, loss_mask = self.batch
        single, _, _ = OnlineMTPModel(self.draft)(input_ids, hidden_states, loss_mask)
        multi, corrects, denoms = OnlineMTPModel(
            self.draft, num_speculative_steps=3, step_weights=[1.0, 0.0, 0.0]
        )(input_ids, hidden_states, loss_mask)
        torch.testing.assert_close(multi, single, rtol=1e-5, atol=1e-6)
        self.assertEqual(3, len(corrects))
        self.assertEqual(3, len(denoms))

    def test_k_step_feeds_previous_draft_output_and_shifts(self):
        calls = []

        class _RecordingDraft(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(pad_token_id=0)
                self.mtp = SimpleNamespace(lm_head=torch.nn.Linear(8, 32, bias=False))

            def forward_hidden(self, input_ids, hidden_states, attention_mask=None,
                               position_ids=None):
                calls.append((input_ids.clone(), hidden_states.clone(), position_ids.clone()))
                return hidden_states + 1.0

        model = OnlineMTPModel(_RecordingDraft(), num_speculative_steps=3)
        input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        model(input_ids, torch.zeros(1, 6, 8), torch.ones(1, 6),
              position_ids=torch.tensor([[4, 5, 6, 7, 8, 9]]))
        self.assertEqual(3, len(calls))
        for k, (ids, hidden, pos) in enumerate(calls):
            # step k sees x[t+k+1] at its own position, fused with step k-1's output
            self.assertEqual(input_ids[0, k + 1:].tolist(), ids[0, : 5 - k].tolist())
            self.assertEqual(list(range(5 + k, 11 + k)), pos[0].tolist())
            self.assertTrue(torch.all(hidden == float(k)))

    def test_k_step_labels_and_masks(self):
        model = OnlineMTPModel(Qwen3_5MTPDraftModel(_tiny_config()))
        labels, mask = model._shift_for_next_token(
            torch.tensor([[10, 11, 12, 13, 14]]), torch.ones(1, 5), offset=4
        )
        # step 2 (0-based) predicts x_{t+4}: x_4 then ignored padding
        self.assertEqual([14, -100, -100, -100], labels[0].tolist())
        self.assertEqual([1, 0, 0, 0], mask[0].tolist())

    def test_k_step_backward_is_finite_and_reaches_all_trainable_params(self):
        input_ids, hidden_states, loss_mask = self.batch
        self.draft.embed_tokens.weight.requires_grad_(False)
        model = OnlineMTPModel(self.draft, num_speculative_steps=5)
        loss, corrects, _ = model(input_ids, hidden_states, loss_mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, p in self.draft.mtp.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertEqual(5, len(corrects))

    def test_step_weights_length_must_match(self):
        with self.assertRaisesRegex(ValueError, "step_weights"):
            OnlineMTPModel(self.draft, num_speculative_steps=3, step_weights=[1.0])


class ServingRecursionTest(unittest.TestCase):
    """chain_context="prefix": the SGLang EAGLE/NEXTN draft recursion."""

    def _serving_reference(self, draft, input_ids, hidden_states, t, steps):
        """Emulate serving for one anchor t (batch 1): draft-extend writes the
        step-1 entries of the prefix; each later draft step appends one row
        (next token + previous draft output) one position further, with plain
        causal attention over the draft's KV cache."""
        ids = input_ids[:, 1 : t + 2]  # x[1..t+1] fused with h[0..t]
        hid = hidden_states[:, : t + 1]
        outs = []
        for k in range(steps):
            pos = torch.arange(1, ids.shape[1] + 1).unsqueeze(0)
            out = draft.forward_hidden(ids, hid, position_ids=pos)[:, -1:]
            outs.append(out[0, 0])
            if t + 2 + k >= input_ids.shape[1]:
                break
            ids = torch.cat([ids, input_ids[:, t + 2 + k : t + 3 + k]], dim=1)
            hid = torch.cat([hid, out], dim=1)
        return outs

    def test_chain_steps_match_per_anchor_serving_emulation(self):
        for impl in ("eager", "sdpa"):
            with self.subTest(impl=impl):
                torch.manual_seed(0)
                config = _tiny_config()
                config._attn_implementation = impl
                draft = Qwen3_5MTPDraftModel(config).eval()
                seq_len, steps = 14, 4
                input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
                hidden = torch.randn(1, seq_len, config.hidden_size)
                pad = config.pad_token_id
                cache, current, per_step = {}, hidden, []
                with torch.no_grad():
                    for k in range(steps):
                        shift = k + 1
                        ids = torch.nn.functional.pad(
                            input_ids[:, shift:], (0, shift), value=pad
                        )
                        pos = torch.arange(shift, seq_len + shift).unsqueeze(0)
                        current = draft.forward_hidden(
                            ids, current, position_ids=pos, chain_cache=cache
                        )
                        per_step.append(current)
                    for t in (0, 3, 7, seq_len - steps - 1):
                        ref = self._serving_reference(draft, input_ids, hidden, t, steps)
                        for k, expected in enumerate(ref):
                            torch.testing.assert_close(
                                per_step[k][0, t], expected, atol=2e-5, rtol=2e-5
                            )

    def test_chain_step_ignores_other_anchors_chain_entries(self):
        torch.manual_seed(0)
        draft = Qwen3_5MTPDraftModel(_tiny_config()).eval()
        ids = torch.randint(0, 128, (1, 12))
        hidden = torch.randn(1, 12, 64)
        pos = torch.arange(1, 13).unsqueeze(0)

        def second_step(prev):
            cache = {}
            with torch.no_grad():
                draft.forward_hidden(ids, hidden, position_ids=pos, chain_cache=cache)
                return draft.forward_hidden(
                    ids, prev, position_ids=pos + 1, chain_cache=cache
                )

        prev = torch.randn(1, 12, 64)
        changed = prev.clone()
        changed[:, 5] += 1.0
        a, b = second_step(prev), second_step(changed)
        keep = [i for i in range(12) if i != 5]
        torch.testing.assert_close(a[:, keep], b[:, keep])
        self.assertFalse(torch.allclose(a[:, 5], b[:, 5]))

    def test_prefix_context_first_step_equals_single_step(self):
        torch.manual_seed(0)
        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        batch = _tiny_batch(config, seq_len=20)
        single, _, _ = OnlineMTPModel(draft)(*batch)
        multi, corrects, _ = OnlineMTPModel(
            draft,
            num_speculative_steps=3,
            step_weights=[1.0, 0.0, 0.0],
            chain_context="prefix",
        )(*batch)
        torch.testing.assert_close(multi, single, rtol=1e-5, atol=1e-6)
        self.assertEqual(3, len(corrects))

    def test_prefix_context_backward_is_finite(self):
        torch.manual_seed(0)
        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        loss, _, _ = OnlineMTPModel(
            draft, num_speculative_steps=5, chain_context="prefix"
        )(*_tiny_batch(config, seq_len=20))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, p in draft.mtp.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_chain_context_rejects_unknown(self):
        with self.assertRaisesRegex(ValueError, "chain_context"):
            OnlineMTPModel(Qwen3_5MTPDraftModel(_tiny_config()), chain_context="x")


class CausalityTest(unittest.TestCase):
    def test_draft_output_does_not_depend_on_future_positions(self):
        for impl in ("eager", "sdpa"):
            with self.subTest(impl=impl):
                torch.manual_seed(0)
                config = _tiny_config()
                config._attn_implementation = impl
                draft = Qwen3_5MTPDraftModel(config).eval()
                ids = torch.randint(0, config.vocab_size, (2, 16))
                hidden = torch.randn(2, 16, config.hidden_size)
                ids2, hidden2 = ids.clone(), hidden.clone()
                ids2[:, 10:] = 5
                hidden2[:, 10:] = 0
                with torch.no_grad():
                    a = draft.forward_hidden(ids, hidden)
                    b = draft.forward_hidden(ids2, hidden2)
                torch.testing.assert_close(a[:, :10], b[:, :10], atol=1e-5, rtol=1e-5)
                self.assertFalse(torch.allclose(a[:, 10:], b[:, 10:]))


class RopeConfigTest(unittest.TestCase):
    def test_rope_theta_read_from_v5_rope_parameters(self):
        draft = Qwen3_5MTPDraftModel(_tiny_config(rope_theta=10_000_000))
        inv_freq = draft.mtp.rotary_emb.inv_freq
        dim = int(16 * 0.25)
        expected = 1.0 / (10_000_000 ** (torch.arange(0, dim, 2).float() / dim))
        torch.testing.assert_close(inv_freq, expected)


class MTPTrainStrategyTest(unittest.TestCase):
    def test_forward_loss_adapts_model_outputs(self):
        loss = torch.tensor(1.5, requires_grad=True)
        corrects = [torch.tensor([[1.0, 0.0, 1.0]])]
        denoms = [torch.tensor([[1.0, 1.0, 1.0]])]

        class _Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(1, 1)

            def forward(self, input_ids, hidden_states, loss_mask):
                return loss, corrects, denoms

        strategy = MTPTrainStrategy(_Stub())
        batch = SimpleNamespace(
            tensors={
                "input_ids": torch.zeros(1, 3, dtype=torch.long),
                "loss_mask": torch.ones(1, 3),
                "target_last_hidden_states": torch.zeros(1, 3, 4),
            }
        )

        out = strategy.forward_loss(batch)

        self.assertIs(out.loss, loss)
        self.assertAlmostEqual(2.0 / 3.0, out.metrics["accuracy"].item())
        self.assertEqual(3.0, out.metrics["accuracy_denom"].item())
        num, den = out.ratio_metrics["accuracy"]
        self.assertEqual(2.0, num.item())
        self.assertEqual(3.0, den.item())
        loss_num, loss_den = out.loss_terms
        self.assertIsNotNone(loss_num.grad_fn)
        self.assertEqual(4.5, loss_num.item())
        self.assertEqual(3.0, loss_den.item())

    def test_forward_loss_rejects_missing_features(self):
        strategy = MTPTrainStrategy(torch.nn.Linear(1, 1))
        batch = SimpleNamespace(tensors={"input_ids": torch.zeros(1, 3)})
        with self.assertRaisesRegex(ValueError, "missing required features"):
            strategy.forward_loss(batch)

    def test_checkpoint_state_filter_strips_draft_prefix(self):
        strategy = MTPTrainStrategy(torch.nn.Linear(1, 1))
        state = {
            "draft_model.mtp.fc.weight": torch.zeros(2, 2),
            "draft_model.embed_tokens.weight": torch.zeros(4, 2),
            "other.weight": torch.zeros(1),
        }
        filtered = strategy.checkpoint_state_filter(state)
        self.assertEqual({"mtp.fc.weight", "embed_tokens.weight"}, set(filtered))


class NativeMTPInitTest(unittest.TestCase):
    def test_loads_native_mtp_weights_from_target_checkpoint(self):
        from safetensors.torch import save_file

        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        replacement = torch.ones_like(draft.mtp.fc.weight)
        native_state = {
            key: torch.ones_like(value)
            for key, value in draft.native_state_dict().items()
            if key in draft.required_native_state_keys()
        }

        with tempfile.TemporaryDirectory(prefix="mtp-native-init-") as tmpdir:
            save_file(native_state, f"{tmpdir}/model.safetensors")
            _init_from_native_mtp(_cfg(tmpdir), draft)

        self.assertTrue(torch.equal(draft.mtp.fc.weight, replacement))

    def test_partial_native_weights_raise(self):
        from safetensors.torch import save_file

        draft = Qwen3_5MTPDraftModel(_tiny_config())
        with tempfile.TemporaryDirectory(prefix="mtp-native-init-") as tmpdir:
            save_file(
                {"mtp.fc.weight": torch.ones_like(draft.mtp.fc.weight)},
                f"{tmpdir}/model.safetensors",
            )
            with self.assertRaisesRegex(RuntimeError, "missing required native"):
                _init_from_native_mtp(_cfg(tmpdir), draft)

    def test_missing_native_weights_raise_by_default(self):
        draft = Qwen3_5MTPDraftModel(_tiny_config())
        with tempfile.TemporaryDirectory(prefix="mtp-native-init-") as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "no native mtp"):
                _init_from_native_mtp(_cfg(tmpdir), draft)

    def test_missing_native_weights_allowed_for_warm_start(self):
        draft = Qwen3_5MTPDraftModel(_tiny_config())
        before = draft.mtp.fc.weight.detach().clone()
        with tempfile.TemporaryDirectory(prefix="mtp-native-init-") as tmpdir:
            # warm start from a trained draft checkpoint skips the strict check
            _init_from_native_mtp(
                _cfg(tmpdir, draft_checkpoint_path="some/ckpt"), draft
            )
        self.assertTrue(torch.equal(draft.mtp.fc.weight, before))

    def test_native_init_covers_all_mtp_parameters(self):
        """A native checkpoint shipping the full mtp.* key set must overwrite
        every draft native parameter — none may keep its random init."""
        from safetensors.torch import save_file

        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        native_keys = draft.required_native_state_keys()
        replacement = {
            key: torch.ones_like(value)
            for key, value in draft.native_state_dict().items()
            if key in native_keys
        }

        with tempfile.TemporaryDirectory(prefix="mtp-native-init-") as tmpdir:
            save_file(replacement, f"{tmpdir}/model.safetensors")
            _init_from_native_mtp(_cfg(tmpdir), draft)

        after = draft.native_state_dict()
        self.assertTrue(native_keys.issubset(after))
        for key in native_keys:
            value = after[key]
            self.assertTrue(
                torch.all(value == 1), f"{key} was not loaded from native weights"
            )

    def test_native_init_tolerates_merged_checkpoint_keys(self):
        """A previously merged checkpoint carries backfilled shared embeddings
        (mtp.embed_tokens.weight / mtp.lm_head.weight); re-finetuning it must
        not be rejected as an incompatible native state."""
        from safetensors.torch import save_file

        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        state = {
            key: torch.ones_like(value)
            for key, value in draft.native_state_dict().items()
            if key in draft.required_native_state_keys()
        }
        state["mtp.embed_tokens.weight"] = torch.randn(
            config.vocab_size, config.hidden_size
        )
        state["mtp.lm_head.weight"] = torch.randn(config.vocab_size, config.hidden_size)

        with tempfile.TemporaryDirectory(prefix="mtp-native-init-") as tmpdir:
            save_file(state, f"{tmpdir}/model.safetensors")
            _init_from_native_mtp(_cfg(tmpdir), draft)  # must not raise

        self.assertTrue(torch.all(draft.mtp.fc.weight == 1))


class DraftBaseContractTest(unittest.TestCase):
    def test_share_target_embeddings_freezes_and_shares(self):
        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        embed_w = torch.nn.Parameter(torch.randn(config.vocab_size, config.hidden_size))
        head_w = torch.nn.Parameter(torch.randn(config.vocab_size, config.hidden_size))

        draft.share_target_embeddings(embed_w, lm_head_weight=head_w)

        self.assertIs(draft.embed_tokens.weight, embed_w)
        self.assertIs(draft.mtp.lm_head.weight, head_w)
        self.assertFalse(draft.embed_tokens.weight.requires_grad)
        self.assertFalse(draft.mtp.lm_head.weight.requires_grad)

    def test_share_lm_head_disabled_keeps_own_head(self):
        config = _tiny_config(mtp_config={"share_lm_head": False})
        draft = Qwen3_5MTPDraftModel(config)
        own_head = draft.mtp.lm_head.weight
        embed_w = torch.nn.Parameter(torch.randn(config.vocab_size, config.hidden_size))

        draft.share_target_embeddings(embed_w)

        self.assertIs(draft.embed_tokens.weight, embed_w)
        self.assertIs(draft.mtp.lm_head.weight, own_head)

    def test_native_state_dict_uses_native_prefix(self):
        draft = Qwen3_5MTPDraftModel(_tiny_config())
        native = draft.native_state_dict()
        self.assertTrue(native)
        self.assertTrue(all(key.startswith(draft.NATIVE_KEY_PREFIX) for key in native))
        self.assertIn("mtp.fc.weight", native)

    def test_required_native_state_respects_shared_lm_head(self):
        shared = Qwen3_5MTPDraftModel(_tiny_config())
        own = Qwen3_5MTPDraftModel(_tiny_config(mtp_config={"share_lm_head": False}))

        self.assertNotIn("mtp.lm_head.weight", shared.required_native_state_keys())
        self.assertIn("mtp.lm_head.weight", own.required_native_state_keys())


class SelectiveCheckpointLoadingTest(unittest.TestCase):
    def test_sharded_selective_loading(self):
        from safetensors.torch import save_file

        from specforge.modeling.target.checkpoint import (
            list_checkpoint_keys,
            load_selected_tensors,
            read_weight_map,
        )

        with tempfile.TemporaryDirectory(prefix="mtp-ckpt-") as tmpdir:
            save_file(
                {
                    "mtp.fc.weight": torch.zeros(4, 4),
                    "model.embed_tokens.weight": torch.zeros(2, 2),
                },
                os.path.join(tmpdir, "model-00001-of-00002.safetensors"),
            )
            save_file(
                {
                    "mtp.norm.weight": torch.ones(4),
                    "lm_head.weight": torch.zeros(2, 2),
                },
                os.path.join(tmpdir, "model-00002-of-00002.safetensors"),
            )
            weight_map = {
                "mtp.fc.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "mtp.norm.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            }
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": weight_map}, f)

            self.assertEqual(weight_map, read_weight_map(tmpdir))
            self.assertEqual(4, len(list_checkpoint_keys(tmpdir)))
            selected = load_selected_tensors(tmpdir, lambda key: key.startswith("mtp."))

        self.assertEqual({"mtp.fc.weight", "mtp.norm.weight"}, set(selected))
        self.assertTrue(torch.equal(selected["mtp.norm.weight"], torch.ones(4)))

    def test_single_file_selective_loading(self):
        from safetensors.torch import save_file

        from specforge.modeling.target.checkpoint import load_selected_tensors

        with tempfile.TemporaryDirectory(prefix="mtp-ckpt-") as tmpdir:
            save_file(
                {"mtp.fc.weight": torch.zeros(4, 4), "other.weight": torch.zeros(1)},
                os.path.join(tmpdir, "model.safetensors"),
            )
            selected = load_selected_tensors(tmpdir, lambda key: key.startswith("mtp."))
        self.assertEqual({"mtp.fc.weight"}, set(selected))


class ExportRoundTripTest(unittest.TestCase):
    """merge_mtp_into_base round trip on synthetic single-file checkpoints."""

    def test_merge_replaces_native_and_copies_embeddings(self):
        from safetensors.torch import save_file

        from specforge.export.mtp import merge_mtp_into_base
        from specforge.modeling.target.checkpoint import load_selected_tensors

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "base")
            draft = os.path.join(tmpdir, "draft")
            out = os.path.join(tmpdir, "out")
            os.makedirs(base)
            os.makedirs(draft)

            base_embed = torch.randn(128, 64)
            stale_native = torch.zeros(64, 128)
            save_file(
                {
                    "model.embed_tokens.weight": base_embed,
                    "mtp.fc.weight": stale_native,
                },
                os.path.join(base, "model.safetensors"),
            )
            with open(os.path.join(base, "config.json"), "w") as f:
                json.dump({"hidden_size": 64, "tie_word_embeddings": True}, f)

            trained = torch.ones(64, 128)
            save_file(
                {"mtp.fc.weight": trained},
                os.path.join(draft, "model.safetensors"),
            )
            with open(os.path.join(draft, "config.json"), "w") as f:
                json.dump(
                    {
                        "architectures": ["Qwen3_5MTPDraftModel"],
                        "hidden_size": 64,
                        "head_dim": 16,
                    },
                    f,
                )

            merge_mtp_into_base(base, draft, out)

            merged = load_selected_tensors(out, lambda _key: True)
            # trained weights replace the stale native ones
            self.assertTrue(torch.equal(merged["mtp.fc.weight"], trained))
            # shared embedding copied into the native namespace
            self.assertTrue(torch.equal(merged["mtp.embed_tokens.weight"], base_embed))
            # base weights untouched
            self.assertTrue(
                torch.equal(merged["model.embed_tokens.weight"], base_embed)
            )
            # config patched with the draft's structural dims
            with open(os.path.join(out, "config.json")) as f:
                merged_config = json.load(f)
            self.assertEqual(16, merged_config["head_dim"])

    def test_merge_accepts_runtime_training_checkpoint(self):
        from safetensors.torch import save_file

        from specforge.export.mtp import merge_mtp_into_base
        from specforge.modeling.target.checkpoint import load_selected_tensors

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "base")
            runtime = os.path.join(tmpdir, "run-step1")
            out = os.path.join(tmpdir, "out")
            draft_config = os.path.join(tmpdir, "draft-config.json")
            os.makedirs(base)
            os.makedirs(runtime)

            base_embed = torch.randn(128, 64)
            save_file(
                {
                    "model.embed_tokens.weight": base_embed,
                    "mtp.fc.weight": torch.zeros(64, 128),
                },
                os.path.join(base, "model.safetensors"),
            )
            with open(os.path.join(base, "config.json"), "w") as f:
                json.dump({"hidden_size": 64, "tie_word_embeddings": True}, f)
            with open(draft_config, "w") as f:
                json.dump(
                    {
                        "architectures": ["Qwen3_5MTPDraftModel"],
                        "hidden_size": 64,
                        "head_dim": 16,
                    },
                    f,
                )

            trained = torch.ones(64, 128)
            torch.save(
                {
                    "strategy": "mtp",
                    "draft_state_dict": {"mtp.fc.weight": trained},
                },
                os.path.join(runtime, "training_state.pt"),
            )

            merge_mtp_into_base(
                base,
                runtime,
                out,
                draft_config_path=draft_config,
            )

            merged = load_selected_tensors(out, lambda _key: True)
            self.assertTrue(torch.equal(merged["mtp.fc.weight"], trained))
            self.assertTrue(torch.equal(merged["mtp.embed_tokens.weight"], base_embed))
            with open(os.path.join(out, "config.json")) as f:
                merged_config = json.load(f)
            self.assertEqual(16, merged_config["head_dim"])

    def test_runtime_checkpoint_requires_draft_config(self):
        from specforge.export.mtp import merge_mtp_into_base

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = os.path.join(tmpdir, "run-step1")
            os.makedirs(runtime)
            torch.save(
                {
                    "strategy": "mtp",
                    "draft_state_dict": {"mtp.fc.weight": torch.ones(1)},
                },
                os.path.join(runtime, "training_state.pt"),
            )

            with self.assertRaisesRegex(ValueError, "draft_config_path is required"):
                merge_mtp_into_base("unused", runtime, os.path.join(tmpdir, "out"))

    def test_merge_runtime_checkpoint_with_shared_tied_weights(self):
        """Regression: a tied target shares one storage between the draft's
        embed_tokens.weight and mtp.lm_head.weight; safetensors must not choke
        on the aliased pair when writing the merged checkpoint."""
        from safetensors.torch import save_file

        from specforge.export.mtp import merge_mtp_into_base
        from specforge.modeling.target.checkpoint import load_selected_tensors

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "base")
            runtime = os.path.join(tmpdir, "run-step1")
            out = os.path.join(tmpdir, "out")
            draft_config = os.path.join(tmpdir, "draft-config.json")
            os.makedirs(base)
            os.makedirs(runtime)

            save_file(
                {"model.embed_tokens.weight": torch.randn(128, 64)},
                os.path.join(base, "model.safetensors"),
            )
            with open(os.path.join(base, "config.json"), "w") as f:
                json.dump({"hidden_size": 64, "tie_word_embeddings": True}, f)
            with open(draft_config, "w") as f:
                json.dump(
                    {
                        "architectures": ["Qwen3_5MTPDraftModel"],
                        "hidden_size": 64,
                        "head_dim": 16,
                    },
                    f,
                )

            shared = torch.randn(128, 64)
            trained = torch.ones(64, 128)
            torch.save(
                {
                    "strategy": "mtp",
                    "draft_state_dict": {
                        "embed_tokens.weight": shared,
                        "mtp.lm_head.weight": shared,
                        "mtp.fc.weight": trained,
                    },
                },
                os.path.join(runtime, "training_state.pt"),
            )

            merge_mtp_into_base(base, runtime, out, draft_config_path=draft_config)

            merged = load_selected_tensors(out, lambda _key: True)
            self.assertTrue(torch.equal(merged["mtp.embed_tokens.weight"], shared))
            self.assertTrue(torch.equal(merged["mtp.lm_head.weight"], shared))
            self.assertTrue(torch.equal(merged["mtp.fc.weight"], trained))


class MTPRegistrationTest(unittest.TestCase):
    def test_draft_architecture_is_registered(self):
        self.assertIn("Qwen3_5MTPDraftModel", available_drafts())
        self.assertIs(resolve_draft("Qwen3_5MTPDraftModel"), Qwen3_5MTPDraftModel)

    def test_builtin_registry_resolves_mtp(self):
        registration = builtin_algorithm_registry().resolve("mtp")
        self.assertEqual("mtp", registration.spec.name)
        self.assertEqual(
            "Qwen3_5MTPDraftModel",
            registration.providers.model.draft_config.architecture,
        )

    def test_offline_layout_persists_only_final_hidden(self):
        providers = builtin_algorithm_registry().resolve("mtp").providers
        layout = providers.offline_for("text").capture_layout
        self.assertEqual(
            ("input_ids", "loss_mask", "target_last_hidden_states"),
            layout.output_names,
        )
        self.assertIsNone(layout.aux_feature)

    def test_streaming_layout_exposes_final_hidden(self):
        providers = builtin_algorithm_registry().resolve("mtp").providers
        layout = providers.server_streaming_for("text").layout
        self.assertEqual("target_last_hidden_states", layout.last_hidden_feature)


if __name__ == "__main__":
    unittest.main()
