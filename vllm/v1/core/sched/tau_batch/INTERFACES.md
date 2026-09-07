# τ-Batch framework contracts

## Implemented execution boundary

- `ListPackingStrategy.pack(requests, ctx) -> MicroBatchList` sees the whole
  waiting pool. The default packer limits both request count and the sum of
  complete prompt lengths in each micro-batch. If a request does not fit the
  current task, later requests are considered; skipped requests can enter a
  later task. Remaining requests are deferred, never split into prompt chunks.
- `TauBatchPlanner` validates custom packer output. `TauScheduler` checks the
  actual forward token total before allocating KV and again before emitting.
- `TauScheduler.configure_vllm_config(config)` runs during `VllmConfig`
  initialization, before workers/executors. Scheduler construction validates
  the contract without mutating worker-related flags.
- `ListDispatcher.retire(index)` removes a finished/cancelled micro-batch from
  both readiness gates and decode rotation without renumbering in-flight tasks.
- `TauScheduler.pipeline_snapshot()` copies pending requests, active requests,
  per-request context lengths/KV block IDs, prefill readiness, and in-flight
  counts. Requests already admitted but not yet dispatched are excluded from
  the waiting candidate set. `computed_tokens` includes scheduled work, so it
  is not proof that an in-flight KV write has completed.

## Latency predictor injection

`LatencyOracle.predict(features, pp_rank=...) -> LatencyEstimate` predicts one
stage's forward in milliseconds. An oracle is bound to one model, hardware,
PP/TP layout, dtype, backend and measurement scope. Prefill and decode need
separate fitted coefficients. `upper_ms` is an optional calibrated estimate,
not an unconditional SLO guarantee. PP receive/send and queue wait are outside
this prediction; a pipeline estimator accounts for them separately.

For programmatic integration, `TauScheduler(..., planner=planner,
latency_oracle=oracle)` passes the predictor to strategies in `PackContext`.
These two injection arguments are optional and leave the default greedy
strategy unchanged. Parameter loading and fitting are not implemented here;
they should run outside the scheduling hot path. No CLI policy/profile selector
is provided yet.

## Online event/action contract (defined, not yet executed)

`OnlinePolicy.on_event(event, state) -> PlanUpdate | None` is the agreed
interface for arrival, EOS, whole-micro-batch completion, and cancellation.

- Arrival may replace only the undispatched `pending_plan`.
- EOS may propose `RefillProposal(target_index, request_ids)`. The adapter
  must first execute a complete Prefill for new requests; only after its
  completion may they join the target Decode task at an in-flight-safe boundary.
- Whole-micro-batch completion may release capacity and trigger new packing
  while preserving the ceilings and resource commitments of surviving tasks.
- Every proposal is revalidated against current request liveness, available
  token/KV capacity, ceilings and in-flight state before application. A matching
  wave ID alone does not make a proposal current. Existing outputs retain their
  original task identity. Proposals must not mutate active plans directly.
- `PlanUpdate.pending_plan=None` means preserve; an explicit empty plan means
  clear. `None` from the policy means no action.

The fixed-list scheduler currently exposes the state/predictor interfaces but
does **not** invoke `OnlinePolicy` or apply its proposals. Its existing EOS hook
remains notification-only. Pending-plan storage, event delivery and the
prefill-before-refill executor are subsequent adapter work, not an implemented
paper policy. This separation avoids silently accepting actions it cannot run.

## Trace semantics in the current code

Schema 5 records `n`, `s_max`, `s_sum`, `seq_lens`, `tokens`, `phase`, and
`pp_size` on forwards; model/configuration metadata is written separately.
Decode lengths are captured after scheduling advances computed tokens: they
include the current input token. Decode `tokens` counts new input tokens,
whereas `s_sum` sums the full contexts; those are different quantities.

- `emit`/`done`: driver pipeline lifetime, including queueing and result handling.
- `compute`: model-runner host invocation plus device synchronization.
- `recv`: upstream receive/wait; `send`: asynchronous enqueue-return.
- `stage`: raw host `execute_model` envelope, without an extra device barrier.

Plots prefer measured `compute` windows and otherwise show unmodified `stage`
envelopes. Neither plot positioning nor `done.duration_ms / pp_size` supplies
a measured per-stage device time. CUDA/NPU event-based profiling, backend
coverage verification, controlled sampling, regression/validation, run IDs and
TP-rank-aware joins remain profiling work. The detailed phase hooks currently
live in `gpu_worker.py`; an overridden worker must instrument its own path.

## Deliberately unchanged behavior in this patch

The running-count overflow/drop policy, zero-token completion/control-message
handling, and `min_waiting` behavior are unchanged. They remain distinct from
the per-forward token budget and the cancelled-micro-batch readiness fix.
