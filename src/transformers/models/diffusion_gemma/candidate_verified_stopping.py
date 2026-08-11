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
"""Candidate-aware early-exit stopping criteria for DiffusionGemma.

This module provides [`CandidateVerifiedStoppingCriteria`], a drop-in
[`DiffusionGemmaAdaptiveStopping`] that verifies confidence and argmax stability over a
dynamically-extracted *candidate span* of the canvas (the answer region) rather than over
the whole canvas.

It is an adapted port (Mode 2) of the Confidence-Verified Commit (CVC) component of LATCH
("Where and When to Commit: Candidate-Aware Decoding for Diffusion Language Models",
arXiv:2607.28166). The paper's core contribution -- matching the stop decision to
candidate-scoped evidence instead of coarse whole-region confidence statistics -- is kept
at full fidelity. Its auxiliary component, a per-task deterministic candidate parser, is
substituted here by a pluggable, token-space [`CandidateExtractor`] (a tail window by
default, or a delimiter-based extractor for format-aware answer spans). This keeps the
criterion training-free and tokenizer-free while preserving the candidate-scoped stopping
behaviour. The paper's Block-Wise Early Commit (the "where" axis) is intentionally out of
scope: it accelerates *non-final* blocks, while generation-time stopping here governs the
final-block termination decision.
"""

from typing import Protocol

import torch

from .generation_diffusion_gemma import DiffusionGemmaAdaptiveStopping


class CandidateExtractor(Protocol):
    """
    Extracts the candidate span from a denoiser argmax canvas.

    The candidate span is the set of positions whose stabilization actually warrants
    stopping the per-canvas diffusion loop -- typically the committed answer rather than a
    still-churning reasoning prefix.

    Implementations return a boolean mask of shape `(batch_size, canvas_length)` where
    `True` marks a candidate position. A row that is entirely `False` (no candidate found
    yet) is treated as "not ready to stop".

    Args:
        argmax_canvas (`torch.LongTensor` of shape `(batch_size, canvas_length)`):
            The argmax of the latest denoiser prediction.

    Returns:
        `torch.BoolTensor` of shape `(batch_size, canvas_length)`:
            Mask selecting the candidate positions.
    """

    def __call__(self, argmax_canvas: torch.LongTensor) -> torch.BoolTensor: ...


class TailWindowCandidateExtractor:
    """
    Candidate extractor that selects the last `window` tokens of the canvas.

    A parameter-light default: for short-answer tasks the committed answer typically
    occupies the tail of the canvas, so scoping the stop check to that window avoids
    waiting for an unstable prefix to settle. Set `window` to `0` to fall back to
    whole-canvas behaviour.

    Args:
        window (`int`):
            Number of trailing canvas positions considered part of the candidate span.
            `0` selects the entire canvas.
    """

    def __init__(self, window: int):
        if not isinstance(window, int) or window < 0:
            raise ValueError(f"`window` must be a non-negative integer (got {window})")
        self.window = window

    def __call__(self, argmax_canvas: torch.LongTensor) -> torch.BoolTensor:
        batch_size, canvas_length = argmax_canvas.shape
        mask = torch.zeros((batch_size, canvas_length), dtype=torch.bool, device=argmax_canvas.device)
        window = canvas_length if self.window == 0 else min(self.window, canvas_length)
        if window > 0:
            mask[:, -window:] = True
        return mask


class DelimiterCandidateExtractor:
    """
    Candidate extractor that selects the tokens following the last occurrence of a
    delimiter token (e.g. the id of an "Answer:" marker or a closing code fence).

    This is the token-space, tokenizer-free realization of LATCH's format-aware candidate
    parser: the answer span is whatever follows a task-specific marker, re-extracted at
    every denoising step as the marker position settles.

    Args:
        delimiter_token_id (`int`):
            Token id marking the start of the candidate span. Positions strictly after the
            *last* occurrence of this token in each row form the candidate span.
    """

    def __init__(self, delimiter_token_id: int):
        self.delimiter_token_id = delimiter_token_id

    def __call__(self, argmax_canvas: torch.LongTensor) -> torch.BoolTensor:
        batch_size, canvas_length = argmax_canvas.shape
        is_delimiter = argmax_canvas == self.delimiter_token_id  # [batch, seq]
        has_delimiter = is_delimiter.any(dim=-1)  # [batch]
        # Index of the last delimiter per row = first True from the right of `is_delimiter`.
        last_delimiter_idx = canvas_length - 1 - torch.flip(is_delimiter.int(), dims=[-1]).argmax(dim=-1)  # [batch]
        positions = torch.arange(canvas_length, device=argmax_canvas.device)  # [seq]
        after_delimiter = positions[None, :] > last_delimiter_idx[:, None]  # [batch, seq]
        return after_delimiter & has_delimiter[:, None]


class CandidateVerifiedStoppingCriteria(DiffusionGemmaAdaptiveStopping):
    """
    Candidate-aware early-exit stopping strategy for DiffusionGemma.

    Mirrors [`StableAndConfidentStoppingCriteria`] (stability + confidence) but verifies
    both conditions over the *candidate span* extracted from the canvas rather than over
    the whole canvas. This keeps the stop decision matched to evidence of the right scope:
    a long chain-of-thought prefix may keep churning while the answer span has already
    stabilized, and a whole-canvas check would needlessly keep denoising. A row for which
    the extractor finds no candidate never stops early.

    Adapted from the Confidence-Verified Commit (CVC) mechanism of LATCH (arXiv:2607.28166).
    The paper's per-task deterministic candidate parser is substituted by a pluggable
    [`CandidateExtractor`] operating in token space; the candidate-scoped verification --
    the paper's core contribution -- is preserved.

    Args:
        stability_threshold (`int`):
            Number of consecutive denoising steps the candidate span's argmax must stay
            unchanged before the canvas is considered stable. `0` disables the stability
            check.
        confidence_threshold (`float`):
            Upper bound on the mean per-token entropy of `logits` over the candidate span
            for the canvas to be considered confident.
        candidate_extractor (`CandidateExtractor`, *optional*):
            Callable returning the candidate-span mask for the current argmax canvas. If
            omitted, a [`TailWindowCandidateExtractor`] of size `candidate_window` is used;
            pass a [`DelimiterCandidateExtractor`] for format-aware spans.
        candidate_window (`int`, *optional*, defaults to 64):
            Size of the tail window used by the default extractor. Ignored when
            `candidate_extractor` is provided.
    """

    def __init__(
        self,
        stability_threshold: int,
        confidence_threshold: float,
        candidate_extractor: CandidateExtractor | None = None,
        candidate_window: int = 64,
    ):
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.candidate_extractor = candidate_extractor or TailWindowCandidateExtractor(window=candidate_window)
        self.argmax_canvas_history: torch.LongTensor | None = None

    def __call__(self, argmax_canvas: torch.LongTensor, logits: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        """
        Applies the candidate-verified stopping strategy, returning a boolean tensor
        indicating whether to stop for each sample in the batch.

        Args:
            argmax_canvas (`torch.LongTensor`):
                The argmax of the latest denoiser prediction, shape `(batch_size, canvas_length)`.
            logits (`torch.FloatTensor`):
                The predicted logits, after applying logits processors, of shape
                `(batch_size, canvas_length, vocab_size)`.

        Returns:
            `torch.BoolTensor`: A boolean tensor of shape `(batch_size,)` indicating
            whether to stop. Rows with no extracted candidate never stop.
        """
        candidate_mask = self.candidate_extractor(argmax_canvas)  # [batch, seq]
        has_candidate = candidate_mask.any(dim=-1)  # [batch]

        # 1. Stability over the candidate span only: non-candidate positions are treated as
        #    "matching" so they cannot keep the canvas from stabilizing.
        if self.stability_threshold == 0:
            stable = torch.ones(logits.shape[0], device=logits.device, dtype=torch.bool)
        else:
            if self.argmax_canvas_history is None:
                self.argmax_canvas_history = torch.full(
                    (self.stability_threshold, argmax_canvas.shape[0], argmax_canvas.shape[1]),
                    -1,
                    dtype=argmax_canvas.dtype,
                    device=argmax_canvas.device,
                )
            matches = self.argmax_canvas_history == argmax_canvas[None, :, :]  # [T, batch, seq]
            matches = matches | ~candidate_mask[None, :, :]
            stable = matches.all(dim=-1).all(dim=0)
            self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, shifts=-1, dims=0)
            self.argmax_canvas_history[-1] = argmax_canvas

        # 2. Confidence over the candidate span only (masked mean of per-token entropy).
        dist = torch.distributions.Categorical(logits=logits)
        token_entropy = dist.entropy()  # [batch, seq]
        mask = candidate_mask.to(token_entropy.dtype)
        entropy_sum = (token_entropy * mask).sum(dim=-1)  # [batch]
        candidate_count = mask.sum(dim=-1).clamp(min=1.0)  # [batch]
        confident = (entropy_sum / candidate_count) < self.confidence_threshold

        return stable & confident & has_candidate

    def reset(self):
        self.argmax_canvas_history = None
