# Experimental Ray PP across 910B and 310P

This is an opt-in **implementation for hardware bring-up**, not a claim that
mixed-SoC model inference has passed. All new code is in the vLLM repository.
The default executors, existing CPU transport, Ascend plugin sources, Ray,
CANN and drivers are unchanged. Device compute still uses the installed
vLLM-Ascend workers; each host needs a compatible build for its own SoC.

## Supported first implementation

One scheduler and model, four consecutive PP stages, TP=DP=PCP=DCP=1.
Ranks 0/1 live on one 910B host and ranks 2/3 on one 310P host. Each actor owns
one Ray `NPU` resource and uses logical device 0 in its isolated device view.
An explicit `validation_mode: "910b_only"` plan permits four 910B stages on
two nodes for bring-up. It does not validate the 310P worker or mixed-SoC math.
Actual device allocations are sorted per node before ranks are assigned; loaded
layer ranges are checked against the plan. Layer counts must be positive and
sum to the checkpoint's transformer layers. Embedding and output head remain
at the native first/last stage; layer counts do not account for their cost.

Only dense text LlamaForCausalLM / Qwen2ForCausalLM, FP16, eager, MRv1 and
synchronous scheduling are admitted. Speculation, KV connectors/PD, LoRA,
quantization, multimodal, async scheduling, DBO and external worker extensions
and custom additional_config are rejected. The exact automatically inserted
Ascend 0.23 eager defaults are admitted; enabling allreduce fusion or adding
unknown configuration is still rejected. Single in-flight scheduler batch is intentional: it establishes
the dependency-correct execution path before throughput-oriented overlap.
This version does not promise to saturate four devices at once.

## Execution and dependencies

- `config.py` validates topology and runtime before actors are created.
- `HeteroPPExecutor` connects to an **existing** Ray cluster and reserves one
  NPU and one CPU per actor with hard node affinity. All rank RPCs are posted
  before the driver waits. Waiting for a receiving actor before posting the
  sending actor would deadlock. Every result is gathered; middle-stage failure
  poisons the executor and kills only its own actors, without restarting them.
- `actor.py` selects the local plugin's hardware profile after Ray allocates
  the device. A 310P plan cannot silently instantiate a 910B worker. Each actor
  uses the local ACL SoC name on older plugins without the profile API, and
  adapts to the core's worker-wrapper constructor. RPC dispatch uses the same
  `run_method` API as the native executor. Each actor
  deep-copies the shared config, loads native layers, and reports actual device,
  layer range, software versions, model-config and adapter source hashes.
- `worker.py` lazily subclasses the local native worker. It replaces distributed
  initialization with CPU/Gloo WORLD and model groups, and uses the native
  model runner to execute the scheduled layers. It **does not** use the ordinary
  Ray compiled DAG's direct NPU tensor serialization path.
- `groups.py` installs an actor-local group constructor before WORLD exists.
  It uses the core Gloo group constructor without a device communicator. This
  bypasses Ascend's HCCL-specific group constructor without editing it or
  monkey-patching torch.distributed. TP1 and other singleton model groups
  perform no device collective. PP dictionary broadcast/return paths use the
  staged transport; runner broadcast_pp_output is explicitly rejected because
  it has its own stage-send protocol.
- `channel.py` sends dictionary keys, dtype, logical shape and step sequence
  as control metadata, and uses CpuStagingTransport for values. Metadata uses
  PyTorch object serialization: the Ray/Gloo cluster must be trusted. Peers must
  run the same sources and protocol. A receive completes CPU network IO and
  H2D **before** computation. Send buffers are snapshotted and all send handles
  are waited before the actor returns. Source storage may then be reused.
- Execute and sample are separate RPCs, matching the EngineCore contract. The
  final stage's native output is returned to the scheduler. Async native output
  wrappers are resolved locally, not serialized through Ray.
  Shutdown closes admission and drains already submitted model RPCs before
  destroying groups. A final bookkeeping step can still be pending after
  generate() returns. Repeated/reentrant shutdown does not kill active cleanup.

Failures have a finite RPC and Gloo timeout; failed requests are not retried.
Graceful shutdown drains and destroys groups, then the executor kills its owned
actors. It never calls ray.shutdown on someone else's shared cluster. A network
or rank failure terminates the pipeline; resuming model requests is not provided.

## Deployment

1. Select a **matched** vLLM core / vLLM-Ascend release or verified commit pair
   and port these additions to that pair on both hosts. Do not mix arbitrary
   sibling checkouts. Device/runner method signatures still belong to the plugin
   and may change between versions.
2. Install each host's own SoC-compatible TorchNPU/CANN/plugin packages. Never
   copy 910B-only wheels or the head's library environment onto 310P. Match the
   model/tokenizer/checkpoint weights, core/adapter sources, PyTorch and Ray.
3. Check that each host independently runs the target model's basic operations;
   FP16 eager tiny-model inference is the first gate. Verify the concrete 310P
   board's memory and kernel support. Passing CPU transport tests proves neither.
4. Start Ray in those environments: on A, `ray start --head --node-ip-address=A_IP
   --port=6379`; on B, `ray start --address=A_IP:6379 --node-ip-address=B_IP`.
   Ascend Ray device detection must expose **two NPU resources per node**. Do not
   counterfeit NPU resources on a machine where devices are not detected.
   Reserve at least two free CPU cores per node for the actors as well.
   Set each daemon's GLOO_SOCKET_IFNAME to the reachable interface on that host.
   All necessary Ray ports and the TCP/Gloo store/data connections must be reachable;
   allowing only port 6379 is insufficient. The engine runs on A.
5. Edit `examples/online_serving/hetero_pp/plan.example.json` with actual Ray node
   IPs and feasible layer counts. The example is for **48** transformer layers,
   not a measured optimal split. `192.0.2.*` are documentation placeholders.
   For the 32-layer 910B-only bring-up, use
   `plan.910b-validation.example.json` instead and edit both node IPs.
6. With the updated vLLM on both nodes, launch on A:

   ```bash
   MODEL=/same/path/to/checkpoint PLAN=/path/to/plan.json \
     bash examples/online_serving/hetero_pp/serve.sh
   ```

   Launching initializes devices and loads layers. Do not run on occupied cards.
   This script uses the dedicated Python serving entrypoint, which passes the
   executor class to vLLM's PP admission check. Passing only the class-name string
   to `vllm serve --distributed-executor-backend` is not sufficient on 0.23.
   No software bandwidth wait is injected. Gloo traffic follows the physical
   host network; apply/verify the specified physical bandwidth externally.
7. First check startup reports and actual layer ownership, then one short
   prefill and 16-token decode, then multiple decode steps and request batches.
   Verify full output token sequences/counts against an appropriate homogeneous
   reference. Floating-point differences across SoCs need explicit numerical
   analysis; do not weaken gates silently. Report host-copy/network/H2D costs,
   memory usage and end-to-end latency separately before optimizing overlap.

## Verification scope

CPU unit tests: topology/admission, local selection, stage completion ordering,
empty steps, final output selection, all-rank failure propagation and finite
Ray waits. CPU/Gloo multiprocess checks cover four-stage hidden/residual values,
noncontiguous tensors, token return, sequence validation and core group API.
The real Ascend worker/model, native SoC operations, two physical hosts and
hardware throughput are **not** covered by these CPU fixtures.

The 2026-11-06 hardware bring-up additionally ran CodeLlama-7B-Instruct on
four real 910B3 cards, split between two Ray container nodes on one physical
host. Both 12/8/7/5 and 5/7/8/12 layer layouts were checked against a single-NPU
reference, using batch sizes 1 and 4 and 16-token continuations. These are
functional smoke checks, not mixed-SoC, physical cross-host network, or
throughput validation. Original failures and corrected-version checks are
retained in the bring-up evidence, not folded into the older research data.

Run `python -m unittest discover -s tests/distributed/hetero_pp -v` in an existing
CPU-capable PyTorch/vLLM environment. No dependency installer is provided.
