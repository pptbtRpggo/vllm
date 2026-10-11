# Ascend software heterogeneity experiments

These experiment workers add device-side delays to native Ascend execution. They do not change NPU compute capability or physical HCCL bandwidth. Leave all heterogeneity environment variables unset to use the native worker.

## Compute simulation

`VLLM_TP_COMPUTE_SCALES` supplies one slowdown factor per TP rank, for example `1,1,2,4`. Factors must be at least one. TP stretches attention/MLP compute before decoder-layer all-reduces using queued NPU timing and wait kernels. It does not stretch embedding or LM head compute. PP stretches stage compute, so the coverage of the two simulations differs. These delays approximate slower devices; they are not measurements of real heterogeneous hardware.

The device-delay shared library is required through `VLLM_ASCEND_DELAY_LIBRARY`. After loading the CANN environment, build it with `PYTHON=python bash benchmarks/build_ascend_delay.sh output/delay_lib`; use a library compatible with the installed TorchNPU and CANN versions. Formal serving does not synchronize the device or use a CPU callback to implement these waits.

## Code paths and timing costs

The Ascend implementation separates execution, device timing and cost aggregation:

- `vllm/v1/worker/pp_ascend_worker.py` preserves native receive → stage compute → send ordering and returns the same output types as the Ascend worker.
- `vllm/distributed/pp_stream.py` adds one device wait per stage/microbatch and records optional layer events. It hands snapshots to the background trace writer, which reads them after their completion event.
- `vllm/v1/worker/tp_ascend_worker.py` uses `tp_hetero.py` to stretch local attention/MLP segments before their collectives. Forward execution and deferred trace resolution are separate methods.
- `vllm/distributed/ascend_device_delay.py` submits device timing/wait kernels. PP uses the direct adapter to preserve its tested ProcessGroup ordering; TP uses queued TorchNPU operators. These adapters are both needed.
- `vllm/distributed/pp_layer_cost.py` validates trace timing, then builds representative layer, endpoint, fixed-overhead and measured link costs. `pp_partition.py` consumes these costs subject to memory bounds.

For an Ascend PP stage, let `B` be its pre-wait interval, `D` its observed extra wait and `L_i` a layer event interval. The stored layer cost is `L_i * (B + D) / B`. Embedding, final norm and LM head use the same proportional attribution; the remainder is stored as runner overhead. DP must not multiply these stored costs by the slowdown factor again. These are elapsed device-stream intervals, which may include execution gaps; they are not sums of kernel active times.

By default, cost aggregation first averages local layers within each microbatch and then averages all valid microbatches, including mixed prefill/decode steps. The optional `phase-balanced` aggregation instead averages the pure prefill and pure decode means equally and excludes mixed steps. Neither aggregation predicts serving P99 TTFT or request queueing. See [PP profiling](../usage/pp_profiling.md) for the trace schema and planner options.

The CUDA PP worker still uses synchronous timing helpers in `pp_hetero.py` and `pp_layer_trace.py`. They are separate from the Ascend device-kernel path and remain necessary for CUDA experiments. The retired host-callback prototype and its diagnostic benchmark are available in Git history; they are not part of the serving implementation.

## TP target-total network simulation

First measure the native collectives on the same host and with the same number of devices:

```bash
torchrun --standalone --nproc_per_node=4 benchmarks/tp_network_calibration.py --output output/native_tp4
```

Use two processes for TP2. The calibration uses the installed `NPUCommunicator`, checks collective outputs, and fits a native cost curve for each operation. Synchronization occurs only at standalone measurement boundaries, not in formal serving.

Set `VLLM_TP_CROSS_GROUP_SIZE=2` for two groups of two TP4 ranks. Set `VLLM_TP_CROSS_NETWORK` to JSON with this structure, substituting the measured native curves:

```json
{
  "tp_size": 4,
  "cross_group_size": 2,
  "bandwidth_gbps": 25,
  "native_collectives": {
    "all_reduce": {"bandwidth_gbps": 100, "latency_ms": 0.1},
    "all_gather": {"bandwidth_gbps": 100, "latency_ms": 0.1},
    "reduce_scatter": {"bandwidth_gbps": 100, "latency_ms": 0.1}
  }
}
```

The numbers in this example are placeholders. The calibration topology must match the serving TP size and group split. All three native curves are required. Do not combine this setting with `VLLM_TP_CROSS_EXTRA_BANDWIDTH_GBPS` or `VLLM_TP_CROSS_EXTRA_LATENCY_MS`, which use additive delays.

For logical cross-group payload size `S`, native cost is `L_native + 8*S/(B_native*1e6)` milliseconds. Target cost is `8*S/(B_target*1e6)` milliseconds. Added delay is `max(0, target_cost - native_cost)`, using the same formula as PP target-total simulation. A wait kernel follows the native collective; an already slower native operation cannot be accelerated.

The logical payload is an idealized lower bound across a two-group cut: input bytes for all-reduce, `max(left, right)` times input bytes for all-gather, and `ceil(max(left, right)*input_bytes/TP_size)` for reduce-scatter. It does not describe HCCL routing, contention, or physical wire bandwidth. A 100/25 Gbps parameter must not be reported as a measured physical link rate.

## Paired TP and PP experiments

`benchmarks/tp_pp_compare.py` consumes completed PP experiments, their saved formal request IDs and token IDs, and their frozen SLO thresholds:

```bash
python benchmarks/tp_pp_compare.py --reference-root output/pp_reference --data output/pp_reference/data --model /path/to/model --output output/tp_matched --native-calibration output/native_tp4/native_calibration.json
```

Use `--validate-only` to check inputs without launching a service. The benchmark verifies request order, prompt hashes, input/output lengths, concurrency, warmup samples, and key serving parameters against saved PP results. TP uniformly shards the same model across the same number of devices. Each concurrency point receives independent warmup; formal outputs are not shortened. Services are reused within a model/network configuration.

Successful points are skipped on restart. Failed results are retained before retry. A saved protocol records source and delay-library hashes; changed protocols require a new output directory. The script stops on failed warmup or incomplete formal responses. Comparisons should disclose service reuse and the differing compute-simulation coverage described above.

## PP sensitivity experiments

`benchmarks/pp_sla_experiment.py --max-num-batched-tokens` changes the scheduler token budget (default 2048) in both the saved serving configuration and the actual server command. It is an upper bound on tokens processed in a scheduler step, not a fixed request batch size. Client concurrency remains a separate control.

`benchmarks/pp_sensitivity.py` prepares and runs four one-variable CodeLlama34B sweeps: cross-group bandwidth parameters, scheduler token budget, input length, and client concurrency. Each condition profiles real serving traces, computes a memory-feasible latency partition, and compares it with uniform PP on identical requests. Inputs at different fixed lengths use prefixes of the same long source prompts; warmup, profile, and evaluation prefixes are disjoint. This is a performance workload, not a quality evaluation. Formal runs use 128 requests with 64 output tokens; all arms use the same output length. Saved long-output PP results are not reused as the shortened-output baseline.

Use `--reference-root`, `--source-data`, and `--output` to identify the completed PP reference, its dataset, and a fresh output directory. `--prepare-only` checks and writes the workload without starting a service. Successful points are checked and skipped on resume. Run only after other experiments release the devices. A single run with 128 requests supports a pilot comparison of distributions and P90; P99 remains preliminary and does not establish a stable ranking across repeated runs.
