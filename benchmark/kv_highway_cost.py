# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Analytical cost model for mixing short-convolution and full-attention blocks.

During CPU decode, every full-attention block re-reads the whole key/value cache it
has written so far, so its per-token traffic grows linearly with the context length.
A short-convolution block instead keeps a fixed window (a few timesteps) of state,
so its per-token traffic is constant. `mha_traffic_bytes` / `conv_traffic_bytes`
turn that observation into byte counts for a given architecture, and
`highway_layout` decides which blocks get which mixer.

Adapted from *Daedalus-150M: A Convolution-Attention Hybrid Designed for CPU
Inference* (arXiv:2608.20210), which keeps full attention in 6 of 18 blocks and
reports a speed advantage that "is near zero at an empty context and grows with
length, which is what the mechanism predicts and what a merely leaner model would
not show" -- the slope of decode time against context length is the measurement
that isolates the mechanism.
"""

from dataclasses import dataclass

# Bytes per element in the KV cache, indexed by dtype. Anything absent is treated
# as `bits // 8`.
_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8": 1,
    "int8": 1,
    "uint8": 1,
}


def _dtype_to_bytes(dtype: str) -> int:
    """Return the width in bytes of a dtype given by name (e.g. ``"bfloat16"``)."""
    if dtype in _DTYPE_BYTES:
        return _DTYPE_BYTES[dtype]
    if dtype.startswith("int") or dtype.startswith("uint"):
        return int(dtype.replace("int", "")) // 8
    if dtype.startswith("float"):
        return int(dtype.removeprefix("float")) // 8
    raise ValueError(f"Unrecognised dtype: {dtype}")


@dataclass
class HighwayConfig:
    """Cache-shape parameters of one architecture, independent of the mixer layout.

    Arguments:
        num_layers (`int`):
            Number of decoder blocks.
        num_kv_heads (`int`):
            Number of key/value heads (after GQA grouping).
        head_dim (`int`):
            Dimension of each attention head.
        num_attention_heads (`int`):
            Number of query heads. Defaults to `num_kv_heads` (MHA). Only the
            query projections depend on it, which short convolutions do not have,
            so it is not exposed on this class.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str = "bfloat16"
    # Width of the conv state, in timesteps. Two in the paper: the last input and
    # the one before it. Raising it widens the constant per-layer state but keeps
    # it constant in the context length, which is the property that matters.
    conv_state_timesteps: int = 2
    # Extra state a convolutional block keeps besides the windowed inputs: the
    # gated linear-attention style channels Daedalus interleaves with the
    # convolutions.
    conv_channels: int = 0

    def __post_init__(self):
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.conv_state_timesteps <= 0:
            raise ValueError(
                f"conv_state_timesteps must be positive, got {self.conv_state_timesteps}"
            )
        if self.conv_channels < 0:
            raise ValueError(f"conv_channels cannot be negative, got {self.conv_channels}")
        self.itemsize = _dtype_to_bytes(self.dtype)

    @property
    def kv_bytes_per_layer_per_token(self) -> int:
        """Bytes of K/V cache one full-attention layer appends per decoded token.

        K and V each contribute `num_kv_heads * head_dim` elements.
        """
        return 2 * self.num_kv_heads * self.head_dim * self.itemsize

    @property
    def conv_state_bytes_per_layer(self) -> int:
        """Bytes of state one short-conv layer holds, for any context length.

        Windowed inputs plus the extra convolutional channels, if the
        architecture has them.
        """
        windowed = self.conv_state_timesteps * self.num_kv_heads * self.head_dim * self.itemsize
        extra = self.conv_channels * self.itemsize
        return windowed + extra


# Architectures the benches measure, keyed as the measurement names are keyed.
# The last entry is the paper's own shape: 18 blocks with full attention in 6 of
# them; the others are the T5 family the team just enabled SDPA on.
ARCHITECTURES = {
    "t5-small": HighwayConfig(num_layers=6, num_kv_heads=8, head_dim=64),
    "flan-t5-base": HighwayConfig(num_layers=12, num_kv_heads=12, head_dim=64),
    "hybrid-150m": HighwayConfig(num_layers=18, num_kv_heads=8, head_dim=64, conv_channels=1024),
}


def mha_traffic_bytes(config: HighwayConfig, context_len: int) -> int:
    """Per-token KV cache read traffic of an all-attention model at `context_len`.

    Each layer reads its whole K/V cache, i.e. `2 * context_len` tensors of
    `num_kv_heads * head_dim` elements.
    """
    if context_len < 0:
        raise ValueError(f"context_len cannot be negative, got {context_len}")
    per_layer = 2 * context_len * config.num_kv_heads * config.head_dim * config.itemsize
    return config.num_layers * per_layer


def conv_traffic_bytes(config: HighwayConfig, context_len: int) -> int:
    """Per-token state read traffic of an all-short-conv model at `context_len`.

    Independent of `context_len` by construction -- that is the whole point.
    """
    if context_len < 0:
        raise ValueError(f"context_len cannot be negative, got {context_len}")
    return config.num_layers * config.conv_state_bytes_per_layer


def highway_layout(num_layers: int, num_attention_layers: int) -> list[bool]:
    """Return a per-layer mixer mask: `True` for full attention, `False` for conv.

    The attention blocks are spread evenly over the depth rather than bunched at
    the start: Daedalus keeps full attention in 6 of 18 blocks so that two thirds
    of the network never touches the growing cache, and even spacing keeps the
    global mixing points distributed instead of front-loading them. The last block
    is always attention, so the residual stream is globally mixed just before the
    head reads it.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if not 0 <= num_attention_layers <= num_layers:
        raise ValueError(
            f"num_attention_layers must be in [0, {num_layers}], got {num_attention_layers}"
        )
    # The k-th attention block sits at `round(k * num_layers / num_attention_layers)`
    # for k in [1, num_attention_layers] -- evenly spaced over the depth. `round`
    # rather than `int` keeps the spacing even when the ratio is not exact (e.g.
    # 4 attention blocks in 10 layers), and the last block always lands at
    # `num_layers - 1` so even spacing cannot shift an attention block past the end.
    attention_positions = {
        min(num_layers - 1, round(k * num_layers / num_attention_layers) - 1)
        for k in range(1, num_attention_layers + 1)
    }
    return [i in attention_positions for i in range(num_layers)]


def highway_traffic_bytes(
    config: HighwayConfig, context_len: int, num_attention_layers: int | None = None
) -> int:
    """Per-token KV/state read traffic of a hybrid at `context_len`.

    `num_attention_layers` defaults to one third of the depth, the 6-of-18
    proportion of the paper.
    """
    if context_len < 0:
        raise ValueError(f"context_len cannot be negative, got {context_len}")
    if num_attention_layers is None:
        num_attention_layers = max(1, config.num_layers // 3)
    layout = highway_layout(config.num_layers, num_attention_layers)
    attn = config.kv_bytes_per_layer_per_token * context_len
    per_layer = [attn if is_attn else config.conv_state_bytes_per_layer for is_attn in layout]
    return sum(per_layer)


@dataclass
class DecodeSlope:
    """Traffic vs. context fit for one mixer layout.

    A hybrid with at least one attention block always has a non-zero slope; the
    paper's point is that the slope can be cut in proportion to the number of
    attention blocks, while a merely smaller model scales it down less.
    """

    per_context_byte: float
    constant_bytes: float

    def traffic_at(self, context_len: int) -> float:
        """Predicted per-token read traffic at a given context length."""
        return self.constant_bytes + self.per_context_byte * context_len

    def speedup_at(self, baseline: "DecodeSlope", context_len: int) -> float:
        """Baseline traffic over this layout's traffic, at a given context length."""
        own = self.traffic_at(context_len)
        if own == 0:
            raise ZeroDivisionError("cannot compute a speedup against zero traffic")
        return baseline.traffic_at(context_len) / own

    def slope_ratio(self, baseline: "DecodeSlope") -> float:
        """This layout's slope as a fraction of the baseline's."""
        if baseline.per_context_byte == 0:
            raise ZeroDivisionError("baseline has no context dependence to reduce")
        return self.per_context_byte / baseline.per_context_byte


def decode_slope(
    config: HighwayConfig, num_attention_layers: int | None = None
) -> DecodeSlope:
    """Fit per-token traffic as `constant + slope * context_len` for one layout.

    The slope is what an experiment should measure: it is the part of decode cost
    that grows with context, so it isolates the cache-reread mechanism from model
    size, kernel quality and quantization.
    """
    if num_attention_layers is not None and not 0 <= num_attention_layers <= config.num_layers:
        raise ValueError(
            f"num_attention_layers must be in [0, {config.num_layers}], got {num_attention_layers}"
        )
    if num_attention_layers is None:
        num_attention_layers = max(1, config.num_layers // 3)
    layout = highway_layout(config.num_layers, num_attention_layers)
    num_conv = config.num_layers - num_attention_layers
    return DecodeSlope(
        per_context_byte=float(config.kv_bytes_per_layer_per_token * num_attention_layers),
        constant_bytes=float(config.conv_state_bytes_per_layer * num_conv),
    )


def profile_contexts(
    config: HighwayConfig,
    context_lengths: list[int],
    num_attention_layers: int | None = None,
) -> dict[str, list]:
    """Traffic of the all-attention baseline and the hybrid across `context_lengths`.

    Returns a dict of parallel lists suitable for `MetricsRecorder`/CSV output:
    context lengths, per-layout predicted traffic in bytes, and the analytic
    traffic speedup of the hybrid over the baseline. The speedup rises with
    context length and saturates at `num_layers / num_attention_layers`, because
    the conv blocks' fixed state stops mattering once the KV reads dominate.
    """
    if num_attention_layers is None:
        num_attention_layers = max(1, config.num_layers // 3)
    baseline = decode_slope(config, num_attention_layers=config.num_layers)
    hybrid = decode_slope(config, num_attention_layers=num_attention_layers)
    return {
        "context_lengths": list(context_lengths),
        "mha_traffic_bytes": [mha_traffic_bytes(config, c) for c in context_lengths],
        "hybrid_traffic_bytes": [highway_traffic_bytes(config, c, num_attention_layers) for c in context_lengths],
        "traffic_speedup": [hybrid.speedup_at(baseline, c) for c in context_lengths],
    }


def compare_to_predictions(logger, runs: dict[str, float] | None = None) -> dict[str, float]:
    """Compare measured per-token decode times against the predicted traffic ratio.

    `runs` maps ``f"{arch}.decode_secs.{context_len}"`` to a measured per-token
    decode time in seconds. For every architecture present in both, the measured
    speedup over the all-attention baseline is reported next to the ratio the
    traffic model predicts, so a reader can check the *shape* of the measured
    curve (near-parity at short context, widening with length) rather than trust
    absolute times, which depend on the host.

    Returns the predicted speedup per run key, for tests. Called by
    `benchmarks_entrypoint` at the end of a bench run.
    """
    if not runs:
        logger.info("no decode measurements to compare against the traffic model")
        return {}
    # Group by architecture, keeping the context length each measurement was taken at.
    by_arch: dict[str, dict[int, float]] = {}
    for key, measured in runs.items():
        name, _, context_len = key.rpartition(".")
        if "." not in name or not context_len.isdigit():
            logger.warning(f"skipping unrecognised measurement key: {key} (want '<arch>.decode_secs.<len>')")
            continue
        by_arch.setdefault(name.removesuffix(".decode_secs"), {})[int(context_len)] = measured

    out = {}
    for arch, samples in sorted(by_arch.items()):
        if arch not in ARCHITECTURES:
            logger.warning(f"skipping unknown architecture: {arch}")
            continue
        config = ARCHITECTURES[arch]
        baseline = decode_slope(config, num_attention_layers=config.num_layers)
        hybrid = decode_slope(config)  # the default 1-in-3 layout
        for context_len, measured in sorted(samples.items()):
            predicted = hybrid.speedup_at(baseline, context_len)
            out[f"{arch}.decode_secs.{context_len}"] = predicted
            logger.info(
                f"{arch} @ {context_len} tokens: {measured:.6f}s/token, "
                f"predicted traffic ratio vs all-attention: {predicted:.2f}x"
            )
    return out
