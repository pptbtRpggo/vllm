# PP profiling and heterogeneous communication

`vllm pp-profile` measures an existing pipeline partition and recommends
contiguous decoder layer counts on the same ordered PP ranks. It is a restricted
adaptation of [EdgeShard](https://arxiv.org/html/2405.14371v1), not a complete
implementation of its joint device selection and execution schedule. Memory
feasibility now prunes candidates using explicit per-device memory bounds.

## Communication slowdown

`VLLM_PP_HETERO=1,2/4` requests rank 1 compute slowdown of 2 and hop 0→1
communication slowdown of 4. **Live communication slowdown now requires an
explicit baseline**; previously it multiplied blocking send/recv wall time,
which incorrectly amplified producer/consumer waiting.

Set these variables before starting the workers:

- `VLLM_PP_COMM_BANDWIDTH_GBPS`: comma-separated baseline rates, in decimal
  Gbit/s, indexed by the source PP rank. Each rate describes one TP lane, not
  aggregate bandwidth across all TP ranks. Supply an entry for every slowed hop.
- `VLLM_PP_COMM_LATENCY_MS`: optional comma-separated fixed baseline latency
  per hop, in milliseconds. Missing entries default to zero.

For a wire payload of `S` bytes, bandwidth `B` Gbit/s, latency `L` ms and
slowdown factor `s >= 1`, the model is:

```text
baseline_ms = L + 8 * S / (B * 1_000_000)
extra_ms    = (s - 1) * baseline_ms
```

Workers add `extra_ms` after completing the transfer at each endpoint. The
sender and receiver delay completion of the same transfer; neither multiplies
its measured peer wait. A missing baseline raises an error instead of using
blocking wall time as a bandwidth measurement. Empty tensor payloads add no
delay. This is software delay emulation, not a network traffic shaper, and does
not implement asynchronous compute/communication overlap.

For example, a 1,000,000-byte payload at a baseline of 8 Gbit/s and zero fixed
latency takes 1 ms in this model. Fourfold slowdown adds 3 ms. If the receive
already waited 100 ms for upstream compute, the recorded receive wall time is
104 ms rather than 404 ms.

The following example assumes a **separately calibrated** baseline of 8 Gbit/s
and 0.5 ms; these values are illustrative, not hardware defaults:

```bash
VLLM_PP_HETERO=1,2/4 \
VLLM_PP_COMM_BANDWIDTH_GBPS=8 \
VLLM_PP_COMM_LATENCY_MS=0.5 \
vllm pp-profile <model> --pipeline-parallel-size 2 --trace-dir /tmp/pp-traces \
  --memory-profile /path/to/serving-memory.json
```

Calibrate on the target topology, TP size and relevant activation sizes. The
wire byte count accounts for the PP send-slice/receiver-all-gather optimization
and its per-tensor overrides. Receiver all-gather, serialization and other
overheads are not automatically inferred by the link model; fixed latency may
approximate some costs, but a single rate/latency pair may not fit every size.

The plan JSON and printed serve command retain the baseline environment
variables. They must also be available to every distributed worker.

## Trace fields and replay

- `send_ms` and `recv_ms` contain measured operation wall durations, including
  peer waits, plus the configured added delay. OS sleep overshoot is not measured
  by this accounting. Do not add both as two independent link transfers.
- `send_transfer_ms` is the modeled hop service time, including its slowdown
  but excluding peer waits. The fitter prefers it over `send_ms` when present.
- `send_bytes` and `recv_bytes` count the local TP lane's wire payload.
- `compute_scale` and `comm_scale` record the effective slowdown used during
  collection. Replaying new traces under the same `VLLM_PP_HETERO` does not
  multiply those scales again. A changed environment is treated as a target
  configuration, with costs adjusted relative to the recorded configuration.

Legacy traces without scale metadata are assumed unscaled. Legacy traces
without a link model are rejected by default: `send_ms` can include
backpressure. `--allow-wall-time-comm` explicitly permits an approximate plan
and emits a warning. `rank_costs[].comm_source` records `link_model` or
`wall_time`; mixed sources within one rank are rejected. Such old traces
cannot establish isolated link cost or whether slowdown was already applied;
recollect with the baseline configuration before relying on predictions. Offline `--skip-run` does not retroactively calibrate old traces
from new bandwidth environment variables.

## Effect on the EdgeShard objective

The paper's throughput bottleneck model assumes that communication and
computation can overlap. A PP pipeline can overlap computations on different
stages without overlapping a sender's communication with its own next forward.
These are separate properties.

In this checkout, the new tracing/hetero paths synchronize devices after
communication. Removing that synchronization alone is insufficient to prove
overlap: the ordinary PyTorch NCCL `send()` path also waits on its work handle,
inserting a dependency into the current compute stream.

The default throughput model is now `blocking`. `--no-overlap-comm` remains
an explicit alias for that default. `--overlap-comm` selects the paper's
`ideal_overlap` model for comparison; it does not enable overlap in the worker.
Both the standalone planner and `vllm pp-profile` use these defaults.

For two stages with transfer duration `D`, compute times `C0`/`C1`, sufficient
independent equal batches and no other costs, the period is
`max(C0 + D, C1 + D)`. For a linear rendezvous pipeline with independent links:

```text
stage_occupancy[i] = incoming_transfer[i] + compute[i] + outgoing_transfer[i]
steady_state_cycle = max(stage_occupancy)
```

The first rank has no incoming hop; the final **selected** rank has no outgoing
hop. An interior rank serializes recv, forward and send. Each transfer occupies
both endpoints at the same time; it is not counted as two consecutive transfers.
Latency still sums each hop once. The DP fixes the final PP size before costing
prefixes, so dropping a trailing rank also removes its incoming link. Searching
up to P ranks and N layers in timing-only mode takes O(P^2 N^2) time and
O(P N) DP storage. Memory checks additionally sum candidate layer bounds for
each local TP device.

This cycle formula is exact for the stated deterministic rendezvous model.
To see attainability, let S_i(n) be completion of batch n's transfer on link i.
A periodic schedule S_i(n)=n*T+a_i with
`a_i - a_(i-1) = C_i + D_i` meets the compute-input dependency. The downstream
readiness constraint reduces to `T >= D_i + C_(i+1) + D_(i+1)`, exactly the
next stage's occupancy bound. Endpoint and repeated-send constraints also
fit within the maximum occupancy. Unit tests separately simulate earliest
recv/compute/send events and exhaustively compare small candidate partitions.

This is not a guarantee of measured vLLM throughput: finite in-flight batches,
autoregressive feedback, shared NIC contention, CPU scheduling/metadata,
receiver TP all-gather, and varying batch/context sizes are not modeled.
`predicted_cost_ms` for throughput is a **steady-state microbatch cycle**, not
per-request latency or measured tokens/s. JSON records `cost_model` and
`cost_kind`. Memory checking is required by default; see the memory profile
section below. Timing-only analysis must explicitly opt out.

The 32-layer, 1 ms/layer, 8 ms-link regression now chooses 16/16 with a 24 ms
cycle. The old inbound-only model chose 20/12 and predicted 20 ms, although its
blocking timeline takes 28 ms per batch. With two fixed ranks and a fixed wire
payload, D changes the predicted throughput but is constant across layer splits.

Live profiling through the standard engine factory checks for a calibrated
bandwidth entry for every PP hop before loading the model. To replay older
traces explicitly as an approximation:

```bash
vllm pp-profile --skip-run --trace-dir /tmp/old-pp-traces \
  --allow-wall-time-comm --allow-unchecked-memory --min-pp-size 2 --max-pp-size 2
```

Workload and warmup filters are now strict: no matching samples raises an error
instead of silently using excluded prefill or warmup steps.

To reproduce the paper's overlap assumptions, future work needs:

1. Nonblocking activation transfers, with completion waits deferred until the
   data or its storage is actually needed. An immediate `isend().wait()` on the
   compute stream does not achieve this.
2. Activation buffer lifetimes and multiple buffer slots compatible with CUDA
   graphs; retaining a tensor reference prevents deallocation, not overwrite
   by the next graph replay.
3. Receiving waits before consuming activations, consistent PP/TP ordering,
   and sufficient independent batches in flight. The next token of batch A
   still depends on A finishing the downstream stages and returning its result.
4. Event-based asynchronous profiling that distinguishes link service,
   backpressure and local computation, without forcing a per-step device sync.
5. Hardware timeline validation and predicted-versus-measured token throughput.

The blocking cost-model correction does not implement asynchronous runtime
execution.

## Memory feasibility

The planner now requires a `--memory-profile /path/to/profile.json`, or a
`pp_memory_profile.json` sidecar in the trace directory. Both Python APIs and
CLIs reject missing bounds by default. The explicit `--allow-unchecked-memory`
escape hatch is for timing-only analysis; its output is marked unchecked and
has no generated serve command. An invalid supplied profile is never silently
ignored, even when that flag is present.

For every candidate interval `[start_layer, end_layer)` on every PP/TP device:

```text
required = sum(layer_weights_bytes[start:end])
         + sum(layer_kv_bytes[start:end])
         + runtime_bytes + activation_bytes + workspace_bytes
         + communication_bytes + graph_bytes + safety_margin_bytes
         + first_stage_bytes  (rank 0 only)
         + last_stage_bytes   (last selected rank only)
required <= budget_bytes
```

This check occurs before every DP transition for latency, blocking throughput,
and ideal-overlap throughput, including the one-rank case. All TP ranks must
fit, not only the TP0 timing representative. Dropping a trailing PP rank moves
the head reserve onto the new last rank. A one-rank candidate charges both
endpoint reserves. If no partition fits, planning raises an actionable error
instead of returning the fastest infeasible candidate. Memory costs are not
multiplied by compute or communication slowdown factors.

**These are checks against supplied bounds, not automatic hardware profiling or
an unconditional OOM guarantee.** Existing stage timing cannot establish memory
for layers moved onto a device or for new graph/batch shapes. A trace of a
16-layer shard does not justify dividing its total peak by 16. Collect or
calculate the following for the target serving configuration:

- `budget_bytes`: usable budget on that individual device after accounting for
  other processes and the intended GPU-memory utilization; not total GPU VRAM
  and not the post-KV-allocation free-memory counter.
- `layer_weights_bytes`: a full-model-length vector of local weight/storage bytes
  on this TP device, including actual quantization scales/padding. Do not simply
  divide total checkpoint bytes by TP size; replicated tensors matter.
- `layer_kv_bytes`: a full-model-length vector for the **target serving workload**,
  including allocator block rounding and concurrent requests. For independent
  full-attention requests a conservative starting point is
  `max_num_seqs * ceil(max_model_len / block_size) * local_page_size_bytes` per
  layer, using that rank's actual KVCacheSpec, not total model KV heads. Other
  cache types require their own bounds; do not apply this formula to all models.
- The six reserve fields must conservatively cover every candidate shard/shape:
  persistent runtime allocations, peak activations, workspaces, communication
  buffers, graph pools, and safety headroom. Account for allocator overhead;
  reserves must not omit non-PyTorch allocations.
- `first_stage_bytes` includes embedding/input-side storage; `last_stage_bytes`
  includes final norm/head/output-side storage for **any** device that might
  become last. Keep these out of decoder weight vectors. Tied weights may be
  conservatively double-counted on a one-rank candidate, but not omitted.

All numbers are integer bytes, all device rows and all layer entries are
required, and zero reserves must be explicit. A small synthetic example (four
layers, two PP ranks, TP=1) is:

```json
{
  "version": 1,
  "num_layers": 4,
  "pp_size": 2,
  "tp_size": 1,
  "serving_config": {
    "model": "example-model", "revision": null, "dtype": "float16",
    "quantization": null, "kv_cache_dtype": "auto", "block_size": 16,
    "max_model_len": 128, "max_num_seqs": 2, "max_num_batched_tokens": 32,
    "gpu_memory_utilization": 0.9
  },
  "devices": [
    {
      "pp_rank": 0, "tp_rank": 0, "budget_bytes": 10485760,
      "layer_weights_bytes": [1048576, 1048576, 1048576, 1048576],
      "layer_kv_bytes": [524288, 524288, 524288, 524288],
      "runtime_bytes": 1048576, "activation_bytes": 1048576,
      "workspace_bytes": 1048576, "communication_bytes": 524288,
      "graph_bytes": 1048576, "safety_margin_bytes": 524288,
      "first_stage_bytes": 1048576, "last_stage_bytes": 1048576
    },
    {
      "pp_rank": 1, "tp_rank": 0, "budget_bytes": 10485760,
      "layer_weights_bytes": [1048576, 1048576, 1048576, 1048576],
      "layer_kv_bytes": [524288, 524288, 524288, 524288],
      "runtime_bytes": 1048576, "activation_bytes": 1048576,
      "workspace_bytes": 1048576, "communication_bytes": 524288,
      "graph_bytes": 1048576, "safety_margin_bytes": 524288,
      "first_stage_bytes": 1048576, "last_stage_bytes": 1048576
    }
  ]
}
```

These illustrative values are not model/device defaults. `pp_size` describes
all profiled ranks; every row needs memory vectors for all layers so the planner
can evaluate new assignments. `tp_size` requires a row for each physical TP
rank, even though fitting selects one timing representative per PP rank.

Live profiling validates the profile's model/revision, dtype/quantization, KV
dtype/block size, context/concurrency/batch-token limits, GPU-memory utilization,
and PP/TP dimensions
against the resolved engine config before generation. Fixed KV allocation
overrides (`kv_cache_memory_bytes` or `num_gpu_blocks_override`) are rejected
until the bounds model supports those allocation policies. Offline planning uses the
scope declared in the supplied profile; old traces cannot prove that identity.
New timing traces record TP size and reject a conflicting memory profile.

Plan JSON embeds the original memory profile and per-device `required_bytes`,
`budget_bytes`, `headroom_bytes`, layer range and cost breakdown. A sidecar is
saved for subsequent `--skip-run`. `memory_feasibility_checked=true` means checks
against `supplied_memory_bounds` passed, not that the candidate was loaded and
benchmarked. Generated serve commands retain the declared serving limits and
TP size; keep the same devices, budgets and other runtime settings (attention
backend, graph mode, adapters, etc.) used to establish the reserves. Recalibrate
the bounds after any such change and validate peak usage on hardware.

## Source evidence for the send dependency

In this checkout, `vllm/v1/worker/gpu_worker.py::execute_model` calls the runner,
then `get_pp_group().send_tensor_dict(...)`. Its GPU branch in
`vllm/distributed/parallel_state.py` calls `torch.distributed.send`. The CUDA
requirements pin PyTorch 2.9.0:

- [send calls isend().wait()](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/distributed/distributed_c10d.py#L2304).
- [WorkNCCL::wait calls synchronize](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L737).
- [synchronizeStream blocks the current CUDA stream on the NCCL completion event](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L728).

Thus the next forward on the dependent compute stream cannot execute before
local send completion, although the CPU may return and enqueue work sooner.
This does not mean the receiver must finish its forward or return from its
Python recv call. The profiling path additionally calls `measure_comm`, which
performs device synchronization after send/recv. NCCL evidence alone does not
establish the behavior of an arbitrary HCCL version; the profiling wrapper's
explicit NPU synchronization is a separate, directly visible dependency.

The [v0.29.0 worker](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/worker/gpu_worker.py#L1020)
uses isend but waits previous device send handles at the start of the next step,
before calling the model runner, to avoid overwriting in-flight send buffers.
