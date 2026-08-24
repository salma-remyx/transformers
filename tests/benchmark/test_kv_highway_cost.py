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
import logging
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark"
    ),
)

from kv_highway_cost import (  # noqa: E402
    ARCHITECTURES,
    HighwayConfig,
    conv_traffic_bytes,
    decode_slope,
    highway_layout,
    highway_traffic_bytes,
    mha_traffic_bytes,
    profile_contexts,
)


@pytest.fixture
def daedalus() -> HighwayConfig:
    """The paper's shape: 18 blocks, full attention in 6 of them."""
    return HighwayConfig(num_layers=18, num_kv_heads=8, head_dim=64, dtype="bfloat16")


def test_mha_traffic_grows_linearly_with_context(daedalus):
    assert mha_traffic_bytes(daedalus, 0) == 0
    one = mha_traffic_bytes(daedalus, 1024)
    assert mha_traffic_bytes(daedalus, 2048) == 2 * one
    # K and V, over every layer.
    assert one == 18 * 2 * 1024 * 8 * 64 * 2


def test_conv_traffic_is_constant_in_context(daedalus):
    assert conv_traffic_bytes(daedalus, 0) == conv_traffic_bytes(daedalus, 4096)
    # Two timesteps of state, no more.
    assert conv_traffic_bytes(daedalus, 4096) == 18 * 2 * 8 * 64 * 2


def test_six_of_eighteen_blocks_are_attention(daedalus):
    layout = highway_layout(18, 6)
    assert sum(layout) == 6
    # Two thirds of the network never touches the growing cache.
    assert len(layout) - sum(layout) == 12
    # The attention blocks are spread over the depth rather than bunched at the
    # start, so global mixing points stay distributed.
    positions = [i for i, is_attn in enumerate(layout) if is_attn]
    assert positions == [2, 5, 8, 11, 14, 17]


def test_layout_spreads_attention_for_uneven_ratios():
    # `round`-based spacing keeps the count exact when the ratio is not 1-in-3.
    assert sum(highway_layout(10, 4)) == 4
    assert sum(highway_layout(4, 2)) == 2
    assert sum(highway_layout(3, 1)) == 1
    # Degenerate ends of the range still work.
    assert all(highway_layout(6, 6))
    assert not any(highway_layout(5, 0))


def test_hybrid_traffic_is_between_the_two_extremes(daedalus):
    assert conv_traffic_bytes(daedalus, 2048) < highway_traffic_bytes(daedalus, 2048, 6) < mha_traffic_bytes(
        daedalus, 2048
    )


def test_slope_cuts_with_the_number_of_attention_blocks(daedalus):
    baseline = decode_slope(daedalus, num_attention_layers=18)
    hybrid = decode_slope(daedalus, num_attention_layers=6)
    # A third of the attention blocks leaves a third of the context dependence.
    assert hybrid.slope_ratio(baseline) == pytest.approx(1 / 3)
    # ...at any context length, unlike a constant reduction from a smaller model.
    for context_len in (128, 2048, 16384):
        assert hybrid.speedup_at(baseline, context_len) > 1.0


def test_speedup_grows_with_context(daedalus):
    """The paper's signature: near-parity at empty context, widening with length."""
    baseline = decode_slope(daedalus, num_attention_layers=18)
    hybrid = decode_slope(daedalus, num_attention_layers=6)
    short = hybrid.speedup_at(baseline, 8)
    long = hybrid.speedup_at(baseline, 2048)
    assert short < long


def test_defaults_match_the_paper_proportion(daedalus):
    assert decode_slope(daedalus).per_context_byte == decode_slope(daedalus, num_attention_layers=6).per_context_byte


def test_speedup_curve_rises_with_context():
    """The curve the bench records: low at short context, approaching 3x.

    This is the shape that distinguishes the mechanism from a smaller model.
    """
    profile = profile_contexts(ARCHITECTURES["hybrid-150m"], [8, 128, 2048, 16384])
    speedups = profile["traffic_speedup"]
    assert speedups == sorted(speedups)
    assert speedups[0] < 2.5
    assert speedups[-1] == pytest.approx(3.0, abs=0.05)
    # The predicted traffic ratio always favours the hybrid.
    assert all(s > 1.0 for s in speedups)


def test_invalid_configs_are_rejected():
    with pytest.raises(ValueError):
        HighwayConfig(num_layers=0, num_kv_heads=8, head_dim=64)
    with pytest.raises(ValueError):
        highway_layout(18, 19)
    with pytest.raises(ValueError):
        mha_traffic_bytes(HighwayConfig(num_layers=4, num_kv_heads=8, head_dim=64), -1)


def test_benches_entrypoint_wires_in_the_comparison():
    """The entrypoint that discovers benches also reports the model's prediction."""
    import benchmarks_entrypoint
    from kv_highway_cost import compare_to_predictions

    # The entrypoint imports this symbol lazily at the end of a run, so it must be
    # importable from the entrypoint's own directory on `sys.path`.
    assert callable(compare_to_predictions)
    logger = logging.getLogger("test")
    runs = {
        "hybrid-150m.decode_secs.2048": 0.0042,
        "t5-small.decode_secs.2048": 0.0091,
        "not-an-arch-key": 0.1,
    }
    reported = compare_to_predictions(logger, runs)
    # Malformed keys are dropped, known architectures are reported.
    assert set(reported) == {"hybrid-150m.decode_secs.2048", "t5-small.decode_secs.2048"}
    assert reported["hybrid-150m.decode_secs.2048"] > 1.0


def test_kv_highway_bench_is_discoverable():
    """`benches/kv_highway.py` is auto-discovered by the entrypoint's scanner."""
    import benchmarks_entrypoint

    bench_path = os.path.join(
        os.path.dirname(benchmarks_entrypoint.__file__), "benches", "kv_highway.py"
    )
    module = benchmarks_entrypoint.import_from_path("kv_highway", bench_path)
    assert hasattr(module, "run_benchmark")

    records = []

    class Recorder:
        def initialise_benchmark(self, metadata):
            records.append(metadata)
            return "bench-1"

        def collect_model_measurements(self, benchmark_id, measurements):
            records.append((benchmark_id, measurements))

    module.run_benchmark(
        logging.getLogger("test"), "repo", "branch", "commit", "msg", Recorder()
    )
    # One benchmark per architecture, each carrying measurements.
    assert len(records) == 6
    measurements = [entry[1] for entry in records if isinstance(entry, tuple)]
    assert all(float(v) > 0 for m in measurements for v in m.values())
