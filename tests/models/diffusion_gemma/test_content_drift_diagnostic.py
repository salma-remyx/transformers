# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.models.diffusion_gemma.content_drift_diagnostic import (
        DriftReport,
        content_drift_report,
        pair_agreement,
        run_drift_audit,
        speed_agreement_frontier,
    )
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        DiffusionGemmaGenerationOutput,
    )


@require_torch
class ContentDriftDiagnosticTester(unittest.TestCase):
    """Integration tests for the content-drift diagnostic.

    The diagnostic is exercised against the real ``DiffusionGemmaGenerationOutput``
    type returned by the generation path, and :func:`run_drift_audit` is driven
    through a stand-in model whose ``generate`` returns that real type.
    """

    BATCH = 2
    PROMPT_LEN = 3
    GEN_LEN = 4

    def _output(self, generated, tokens_per_forward):
        prompt = torch.arange(self.PROMPT_LEN).repeat(self.BATCH, 1)
        sequences = torch.cat([prompt, generated], dim=1)
        return DiffusionGemmaGenerationOutput(
            sequences=sequences,
            tokens_per_forward=torch.tensor(tokens_per_forward, dtype=torch.float),
        )

    def test_pair_agreement_exact(self):
        tokens = torch.tensor([5, 6, 7, 8])
        agreement = pair_agreement(tokens, tokens.clone())
        self.assertTrue(agreement.exact_match)
        self.assertEqual(agreement.token_agreement_rate, 1.0)
        self.assertEqual(agreement.n_tokens, 4)

    def test_pair_agreement_length_drift_counts_as_disagreement(self):
        # Accelerated run is one token shorter -> that tail is a disagreement.
        baseline = torch.tensor([5, 6, 7, 8])
        accelerated = torch.tensor([5, 6, 7])
        agreement = pair_agreement(baseline, accelerated)
        self.assertFalse(agreement.exact_match)
        self.assertEqual(agreement.token_agreement_rate, 0.75)

    def test_report_detects_drift_and_speedup(self):
        gen_base = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        gen_drift = gen_base.clone()
        gen_drift[:, -1] += 100  # last generated token differs in every row -> drift
        baseline = self._output(gen_base, tokens_per_forward=[1.0, 1.0])
        accelerated = self._output(gen_drift, tokens_per_forward=[2.0, 2.0])
        report = content_drift_report(baseline, accelerated, prompt_len=self.PROMPT_LEN)

        self.assertEqual(report.n_pairs, self.BATCH)
        self.assertTrue(report.prompt_stable)  # prompt region identical
        self.assertEqual(report.exact_match_rate, 0.0)
        self.assertLess(report.mean_token_agreement, 1.0)
        self.assertTrue(report.drift_detected)
        self.assertAlmostEqual(report.mean_speedup, 2.0)

    def test_report_no_drift_when_identical(self):
        gen = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        baseline = self._output(gen, tokens_per_forward=[1.5, 1.5])
        report = content_drift_report(baseline, baseline, prompt_len=self.PROMPT_LEN)
        self.assertEqual(report.exact_match_rate, 1.0)
        self.assertFalse(report.drift_detected)
        self.assertAlmostEqual(report.mean_speedup, 1.0)

    def test_report_flags_unstable_prompt(self):
        gen = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        baseline = self._output(gen, tokens_per_forward=[1.0, 1.0])
        # Same generated region but a perturbed prompt prefix in the "accelerated" run.
        prompt = torch.arange(self.PROMPT_LEN).repeat(self.BATCH, 1)
        prompt[0, 0] = 999
        accelerated_sequences = torch.cat([prompt, gen], dim=1)
        accelerated = DiffusionGemmaGenerationOutput(
            sequences=accelerated_sequences,
            tokens_per_forward=torch.tensor([1.0, 1.0]),
        )
        report = content_drift_report(baseline, accelerated, prompt_len=self.PROMPT_LEN)
        self.assertFalse(report.prompt_stable)

    def test_run_drift_audit_drives_generate(self):
        """run_drift_audit calls model.generate and reads the real output type."""
        gen_base = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        fake_model = _FakeDiffusionGemma(generated_baseline=gen_base)
        input_ids = torch.arange(self.PROMPT_LEN).repeat(self.BATCH, 1)

        report = run_drift_audit(
            fake_model,
            input_ids,
            baseline_kwargs={"max_denoising_steps": 48},
            accelerated_kwargs={"max_denoising_steps": 24},  # 2x fewer steps
            prompt_len=self.PROMPT_LEN,
        )
        # Halving denoising steps doubles tokens_per_forward (2x speedup) and the
        # fake model flips a token at the lower step budget -> drift is detected.
        self.assertAlmostEqual(report.mean_speedup, 2.0)
        self.assertTrue(report.drift_detected)

    def test_speed_agreement_frontier_recommends_fastest_near_exact(self):
        # Build a frontier directly from DriftReport objects: as the knob
        # decreases (more acceleration), speedup rises and agreement degrades.
        runs = [
            (48, DriftReport(1.0, 1.00, True, False, 1.0, 2)),  # baseline: exact, slow
            (32, DriftReport(1.0, 0.995, True, True, 1.5, 2)),  # near-exact, faster
            (24, DriftReport(0.5, 0.80, True, True, 2.0, 2)),  # drifted
            (16, DriftReport(0.0, 0.60, True, True, 3.0, 2)),  # badly drifted
        ]
        frontier = speed_agreement_frontier(runs)
        self.assertEqual([p.knob for p in frontier.points], [48, 32, 24, 16])  # ascending speedup
        # Recommended = fastest setting that is still near-exact (>= 0.99 exact match).
        self.assertIsNotNone(frontier.recommended)
        self.assertEqual(frontier.recommended.knob, 32)
        # The fastest_near_exact helper agrees with the precomputed recommendation.
        self.assertEqual(frontier.fastest_near_exact().knob, 32)


class _FakeDiffusionGemma:
    """Stand-in for a DiffusionGemma model whose ``generate`` returns the real output type.

    Mimics the paper's dynamics: a smaller ``max_denoising_steps`` budget commits
    more tokens per step (faster, higher ``tokens_per_forward``) but silently
    flips one generated token (content drift), letting the diagnostic detect it.
    """

    def __init__(self, generated_baseline):
        self.generated_baseline = generated_baseline

    def generate(self, input_ids=None, max_denoising_steps=48, return_dict_in_generate=True, **kwargs):
        batch = input_ids.shape[0]
        steps = max_denoising_steps if isinstance(max_denoising_steps, int) else 48
        gen = self.generated_baseline.clone()
        if steps < 48:  # accelerated setting -> drift
            gen[:, -1] = gen[:, -1] + 100
        sequences = torch.cat([input_ids, gen], dim=1)
        tokens_per_forward = torch.full((batch,), gen.shape[1] / steps, dtype=torch.float)
        if not return_dict_in_generate:
            return sequences
        return DiffusionGemmaGenerationOutput(
            sequences=sequences, tokens_per_forward=tokens_per_forward
        )
