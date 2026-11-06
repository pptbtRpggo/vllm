# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Link costs from paired serving traces, never from configured network rates."""

import math
from collections import defaultdict


def measured_transfer_ms(record: dict) -> float:
    value = record.get("send_service_ms")
    if record.get("send_service_source") != "measured_serving_overlap" or value is None:
        raise ValueError(
            "missing measured send_service_ms; "
            f"pairing status: {record.get('send_overlap_status')}; "
            "collect matching serving send/recv traces on a shared host clock. "
            "Network/mock configuration is not a measured communication cost"
        )
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("invalid measured send_service_ms")
    return value


def load_topology_measurements(
    trace_sets, workload, warmup_steps, layer_aggregation="microbatch"
):
    """Pool observed transfers by directed device pair across serving runs.

    Unobserved links remain absent: device DP must not invent their cost.
    Runs must use comparable serving traffic; every observed microbatch has
    equal weight by default, just as in direct layer cost aggregation.
    """
    from vllm.distributed.pp_layer_cost import phase_groups, phase_mean
    from vllm.distributed.pp_partition import _keep_record

    pairs = defaultdict(list)
    for records in trace_sets:
        for rank in range(len(records) - 1):
            source = records[rank][0]["profile_device_id"]
            target = records[rank + 1][0]["profile_device_id"]
            for row in records[rank]:
                if _keep_record(row, workload, warmup_steps):
                    # Fail on invalid observations rather than silently
                    # biasing the mean by dropping failed pairs.
                    measured_transfer_ms(row)
                    pairs[source, target].append(row)
    return {
        pair: phase_mean(
            phase_groups(rows, workload, warmup_steps, layer_aggregation),
            measured_transfer_ms,
        )
        for pair, rows in pairs.items()
    }
