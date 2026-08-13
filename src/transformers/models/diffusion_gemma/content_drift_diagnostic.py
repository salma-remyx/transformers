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
"""Paired diagnostic for content drift in accelerated DiffusionGemma generation.

This module adapts the methodology of *"Faster but Different: Diagnosing and
Controlling Content Drift in Accelerated Multimodal Diffusion Language Models"*
(arXiv:2607.29079) to DiffusionGemma's generation surface. The paper's two
deliverables are (1) a *paired diagnostic* that compares an accelerated run
against the same model's unaccelerated output, and (2) an *implementation-scoped
consistency control* expressed as a monotonic speed--agreement frontier.

This is an **adapted port (Mode 2)**: the core mechanism -- a paired
accelerated-vs-baseline comparison yielding an agreement metric, plus a frontier
sweep over an acceleration knob -- is kept at full fidelity. The paper's
auxiliary components are substituted with target-native equivalents:

* The paper's KV-cache refresh interval is mapped to DiffusionGemma's
  ``max_denoising_steps`` (fewer denoising steps = fewer "refreshes" = faster
  but lower fidelity). The paper's own ``committed tokens per step`` maps
  directly onto ``DiffusionGemmaGenerationOutput.tokens_per_forward``
  (= generated tokens / denoising steps), which we reuse as the speed proxy.
* The paper's human-judged content-substitution audit is replaced by a
  parameter-free token-agreement proxy over the generated region.
* The paper's separate 300-image evaluation harness and blinded annotators are
  intentionally out of scope -- downstream PR territory.

The integration surface is :meth:`DiffusionGemmaGenerationMixin.generate`:
:func:`run_drift_audit` drives ``model.generate`` twice (baseline vs.
accelerated) and consumes the real :class:`DiffusionGemmaGenerationOutput`.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import torch

from .generation_diffusion_gemma import DiffusionGemmaGenerationOutput


@dataclass
class PairAgreement:
    """Token-level agreement for a single baseline/accelerated pair.

    Agreement is measured over the union length: tokens past the shorter
    sequence count as disagreements, so truncation/extension registers as drift.
    """

    exact_match: bool
    token_agreement_rate: float
    n_tokens: int


@dataclass
class DriftReport:
    """Aggregated drift diagnostic across a batch of pairs.

    ``mean_speedup`` is ``accelerated_tpf / baseline_tpf`` (>1 means the
    accelerated run commits more tokens per step and is therefore faster); it is
    ``None`` when ``tokens_per_forward`` is unavailable (e.g. raw tensors).
    """

    exact_match_rate: float
    mean_token_agreement: float
    prompt_stable: bool
    drift_detected: bool
    mean_speedup: float | None
    n_pairs: int


@dataclass
class FrontierPoint:
    """One point on the speed--agreement frontier (one acceleration setting)."""

    knob: float
    speedup: float
    mean_token_agreement: float
    exact_match_rate: float
    n_pairs: int


@dataclass
class SpeedAgreementFrontier:
    """Sorted speed--agreement frontier with a near-exact recommendation."""

    points: list[FrontierPoint] = field(default_factory=list)
    recommended: FrontierPoint | None = None

    def fastest_near_exact(self, min_exact_match_rate: float = 0.99) -> FrontierPoint | None:
        """Fastest knob whose exact-match rate is still >= ``min_exact_match_rate``."""
        candidates = [p for p in self.points if p.exact_match_rate >= min_exact_match_rate]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.speedup)


def _sequences_and_tpf(output: torch.Tensor | DiffusionGemmaGenerationOutput):
    """Normalize an output to ``(sequences, tokens_per_forward or None)``.

    Accepts a :class:`DiffusionGemmaGenerationOutput` (or any object exposing
    ``.sequences`` / ``.tokens_per_forward``) or a raw token tensor.
    """
    if isinstance(output, torch.Tensor):
        return output, None
    sequences = getattr(output, "sequences", None)
    if sequences is None:
        raise TypeError(
            "Expected a DiffusionGemmaGenerationOutput or a torch.Tensor, got "
            f"{type(output).__name__}."
        )
    return sequences, getattr(output, "tokens_per_forward", None)


def _as_batch(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.dim() == 1:
        sequence = sequence.unsqueeze(0)
    return sequence


def pair_agreement(
    baseline_tokens: torch.Tensor, accelerated_tokens: torch.Tensor
) -> PairAgreement:
    """Compute token-level agreement between two 1-D generated-token tensors."""
    if baseline_tokens.dim() != 1 or accelerated_tokens.dim() != 1:
        raise ValueError("pair_agreement expects 1-D token tensors (a single sequence).")

    n_tokens = max(baseline_tokens.shape[0], accelerated_tokens.shape[0])
    if n_tokens == 0:
        # Nothing was generated: trivially agree (and trivially drift-free).
        return PairAgreement(exact_match=True, token_agreement_rate=1.0, n_tokens=0)

    overlap = min(baseline_tokens.shape[0], accelerated_tokens.shape[0])
    matches = int((baseline_tokens[:overlap] == accelerated_tokens[:overlap]).sum().item())
    token_agreement_rate = matches / n_tokens
    exact_match = overlap == n_tokens and matches == n_tokens
    return PairAgreement(
        exact_match=exact_match, token_agreement_rate=token_agreement_rate, n_tokens=n_tokens
    )


def content_drift_report(
    baseline: torch.Tensor | DiffusionGemmaGenerationOutput,
    accelerated: torch.Tensor | DiffusionGemmaGenerationOutput,
    prompt_len: int = 0,
) -> DriftReport:
    """Build a drift diagnostic comparing an accelerated run to a baseline.

    Args:
        baseline: Unaccelerated generation output (or its ``sequences`` tensor).
        accelerated: Accelerated generation output (or its ``sequences`` tensor).
        prompt_len: Number of leading prompt tokens to skip when comparing the
            *generated* region. The skipped prefix is still checked for stability
            (it must be identical across the two runs -- a sanity check that the
            same prompt was fed in).

    The two outputs must have the same batch size.
    """
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be non-negative, got {prompt_len}.")

    b_seq, b_tpf = _sequences_and_tpf(baseline)
    a_seq, a_tpf = _sequences_and_tpf(accelerated)
    b_batch = _as_batch(b_seq)
    a_batch = _as_batch(a_seq)

    if b_batch.shape[0] != a_batch.shape[0]:
        raise ValueError(
            f"Batch size mismatch: baseline has {b_batch.shape[0]} sequences, "
            f"accelerated has {a_batch.shape[0]}."
        )
    n_pairs = b_batch.shape[0]

    exact_matches: list[bool] = []
    agreements: list[float] = []
    prompt_stable = True
    for i in range(n_pairs):
        b_full, a_full = b_batch[i], a_batch[i]
        if prompt_len > 0:
            b_prompt, a_prompt = b_full[:prompt_len], a_full[:prompt_len]
            # A differing prompt length already indicates a setup mismatch.
            prompt_stable = prompt_stable and (
                b_prompt.shape[0] == a_prompt.shape[0]
                and bool(torch.equal(b_prompt, a_prompt))
            )
        agreement = pair_agreement(b_full[prompt_len:], a_full[prompt_len:])
        exact_matches.append(agreement.exact_match)
        agreements.append(agreement.token_agreement_rate)

    exact_match_rate = sum(exact_matches) / n_pairs
    mean_token_agreement = sum(agreements) / n_pairs

    mean_speedup = None
    if b_tpf is not None and a_tpf is not None:
        # tokens_per_forward is per-sequence (shape [batch]) or a scalar; flatten both.
        b_tpf_t = b_tpf.reshape(-1).to(torch.float)
        a_tpf_t = a_tpf.reshape(-1).to(torch.float)
        if b_tpf_t.shape[0] == n_pairs and a_tpf_t.shape[0] == n_pairs:
            # Guard against division by zero for degenerate baselines.
            safe = b_tpf_t != 0
            if bool(safe.all()):
                mean_speedup = float((a_tpf_t[safe] / b_tpf_t[safe]).mean().item())

    return DriftReport(
        exact_match_rate=exact_match_rate,
        mean_token_agreement=mean_token_agreement,
        prompt_stable=prompt_stable,
        drift_detected=mean_token_agreement < 1.0,
        mean_speedup=mean_speedup,
        n_pairs=n_pairs,
    )


def speed_agreement_frontier(
    frontier_runs: Iterable[tuple[float, DriftReport]],
) -> SpeedAgreementFrontier:
    """Build the speed--agreement frontier from a sweep over an acceleration knob.

    Args:
        frontier_runs: Pairs of ``(knob_value, drift_report)`` -- e.g. one entry
            per ``max_denoising_steps`` setting, each report comparing that
            accelerated setting to the same baseline. Reports lacking a speedup
            (``mean_speedup is None``) are skipped, as they carry no speed signal.

    Returns:
        A :class:`SpeedAgreementFrontier` whose ``points`` are sorted by
        ascending speedup, with ``recommended`` set to the fastest near-exact
        point (exact-match rate >= 0.99) -- the paper's "near-exact agreement at
        a measured speedup" recommendation, expressed as a concrete knob value.
    """
    points: list[FrontierPoint] = []
    for knob, report in frontier_runs:
        if report.mean_speedup is None:
            continue
        points.append(
            FrontierPoint(
                knob=float(knob),
                speedup=report.mean_speedup,
                mean_token_agreement=report.mean_token_agreement,
                exact_match_rate=report.exact_match_rate,
                n_pairs=report.n_pairs,
            )
        )
    points.sort(key=lambda p: p.speedup)
    frontier = SpeedAgreementFrontier(points=points)
    frontier.recommended = frontier.fastest_near_exact()
    return frontier


def run_drift_audit(
    model,
    input_ids: torch.Tensor,
    baseline_kwargs: Mapping[str, object] | None = None,
    accelerated_kwargs: Mapping[str, object] | None = None,
    prompt_len: int = 0,
) -> DriftReport:
    """Drive ``model.generate`` twice and return the paired drift diagnostic.

    This is the integration entry point: it calls the existing
    :meth:`DiffusionGemmaGenerationMixin.generate` path (the call site) with a
    baseline and an accelerated parameterization and consumes the real
    :class:`DiffusionGemmaGenerationOutput`.

    Example::

        # Baseline: full denoising budget. Accelerated: halve the denoising steps.
        report = run_drift_audit(
            model,
            input_ids,
            baseline_kwargs={"max_denoising_steps": 48, "return_dict_in_generate": True},
            accelerated_kwargs={"max_denoising_steps": 24, "return_dict_in_generate": True},
            prompt_len=input_ids.shape[1],
        )
        # report.mean_speedup > 1 and report.drift_detected tells you whether
        # the 2x-faster setting silently changed the generated content.
    """
    baseline_kwargs = dict(baseline_kwargs or {})
    accelerated_kwargs = dict(accelerated_kwargs or {})
    # Ensure we get a rich output to read tokens_per_forward from.
    baseline_kwargs.setdefault("return_dict_in_generate", True)
    accelerated_kwargs.setdefault("return_dict_in_generate", True)

    baseline_output = model.generate(input_ids=input_ids, **baseline_kwargs)
    accelerated_output = model.generate(input_ids=input_ids, **accelerated_kwargs)
    return content_drift_report(baseline_output, accelerated_output, prompt_len=prompt_len)
