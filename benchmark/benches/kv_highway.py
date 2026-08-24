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
import os
import sys
from logging import Logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_highway_cost import ARCHITECTURES, profile_contexts

# The context lengths at which the paper reports its decode measurements: the
# speed advantage is near zero at an empty context and grows with length.
CONTEXT_LENGTHS = [128, 512, 1024, 2048, 4096]


def run_benchmark(
    logger: Logger,
    repository: str,
    branch: str,
    commit_id: str,
    commit_msg: str,
    metrics_recorder=None,
    num_tokens_to_generate=100,
):
    """Record the per-token cache-read traffic of an all-attention baseline against
    a short-conv/attention hybrid, across growing context lengths."""
    for name, config in ARCHITECTURES.items():
        profile = profile_contexts(config, CONTEXT_LENGTHS)
        # The suffix on each key is the index into CONTEXT_LENGTHS, so the CSV rows
        # stay self-describing without widening the measurements table.
        measurements = {
            f"{name}.hybrid_traffic_bytes.{i}": float(b)
            for i, b in enumerate(profile["hybrid_traffic_bytes"])
        }
        measurements.update(
            {
                f"{name}.traffic_speedup.{i}": float(s)
                for i, s in enumerate(profile["traffic_speedup"])
            }
        )
        # The slope is the headline number: the share of decode traffic that still
        # grows with context, as a fraction of the all-attention baseline.
        measurements[f"{name}.slope_ratio"] = float(max(1, config.num_layers // 3)) / config.num_layers

        if metrics_recorder is None:
            logger.info(f"{name}: {profile}")
        else:
            benchmark_id = metrics_recorder.initialise_benchmark(
                {"benchmark": "kv_highway", "arch": name}
            )
            metrics_recorder.collect_model_measurements(benchmark_id, measurements)
