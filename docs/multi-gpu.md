# Qwen3.8 tensor parallelism and the SGLang platform adapter

This change targets **one host, multiple visible GPUs, one Qwen3.8-Flash-Next
model**, using the original Hugging Face safetensors checkpoint and NVFP4 routed
experts stored in host RAM. Start with TP=2 on the RTX 5090 platform. TP=4/8 also
need whole query/value heads and whole 16-element NVFP4 scale blocks on each rank.
The shipping expert intermediate size is 640: each rank gets 320/160/80 channels
at TP=2/4/8. Allocate the same number of GPUs to the container as the selected TP.

The implementation and CPU checks do **not** constitute RTX 5090 acceptance or a
performance measurement. Real checkpoint loading, NCCL, CUDA graphs and generated
answers still need to be checked on the target host. No multi-node launcher is
implemented. Multiple nodes should run separate services until that exists.

## Ownership and limits

| Component | Per-rank ownership |
| --- | --- |
| Token embedding and LM head | Vocabulary rows, padded for equal collective sizes |
| QSA attention | Whole query heads, including each head's output gate; KV heads split or replicated |
| QSA indexer | Replicated so every rank selects the same blocks |
| GDN | Whole key/value heads, matching conv channels, recurrence parameters and local state pool |
| Attention/GDN output | Input columns with an all-reduce of the partial outputs |
| Shared expert | Gate/up rows and matching down columns, with an all-reduce |
| Routed NVFP4 experts | Intermediate rows of gate/up and matching packed columns of down; scales follow their weights |
| Router, hyper-connections and PLE | Replicated |

Use `--moe-strategy offload --quant-backend moe.nvfp4=triton` first. Host offload
keeps expert weights in CPU RAM and executes the selected experts on the GPUs.
W4A4, Marlin and b12x expert kernels still reject TP>1. Quantized dense projections
are sliced with their normalized scales, and a cut through a quantization block
is rejected. The first target is bf16 dense projections with NVFP4 routed experts,
as in RadixArk's Qwen3.8 checkpoint. Different requants need their own validation.

TP currently requires `--text-model-only`; vision tower weight sharding and MTP
speculative decoding are explicitly rejected. Use the original safetensors
directory, not a TP=1 FTW export or a GGUF checkpoint, for this profile.

Pinned PLE loads a full table in **each rank's process**. A 47.7 GiB table therefore
uses about 95.4/190.8/381.6 GiB at TP=2/4/8, in addition to expert banks, transient
loads and the OS. These are separate allocations, not a shared table. Start with
`--ple-backend disk` when the host-memory budget is unknown; it reads table rows
from the mounted checkpoint and trades RAM for storage traffic. Benchmark pinned
PLE only after accounting for this replication. Expert banks are sharded; do not
multiply their entire unsharded size by TP when estimating steady-state RAM.

## Launch directly

```bash
ft serve --model /mnt/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name Qwen3.8-Flash-Next-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --text-model-only \
  --moe-strategy offload --quant-backend moe.nvfp4=triton \
  --ple-backend disk --expert-load serial --disable-pynccl \
  --distributed-timeout 1800 \
  --max-running-requests 2 --max-seq-len-override 16384 \
  --max-output-tokens 2048 --max-prefill-length 4096 \
  --kv-reserve-tokens 32768 --memory-ratio 0.90
```

`--disable-pynccl` selects PyTorch NCCL, avoiding the custom PyNCCL JIT extension's
dependency on system NCCL development headers/libraries. It does not disable
NCCL communication. The extended timeout allows ranks to finish cold loading at
different times. It is not a remedy for a dead rank or a failed collective.

Do not change `CUDA_VISIBLE_DEVICES` inside a worker. FreeToken assigns visible
devices to ranks; an optional `--gpu 0,1` selects entries within the container's
allocation. Check the allocation, GPU topology and NCCL connectivity before a
large model load. Do not disable NCCL P2P/SHM/IB globally without a transport error
that justifies it. Size the container's shared memory and host RAM for the actual
allocation; a fixed `/dev/shm` size cannot be derived from GPU count alone.

## Platform image

`Dockerfile.sglang` is an adapter layer for an **already built, patched engine
image**. It installs no SGLang engine, CUDA or PyTorch, and uses no COPY/ADD.
The platform may launch either `python -m sglang.launch_server`, `serve ...`, or
arguments through the image entrypoint. It translates them to `ft serve` and uses
`execve`, preserving worker supervision and container signal handling.

**The supplied image at FreeToken `377a3bc` cannot be used unchanged.** Its loader
rejects TP>1 and it lacks this adapter module. The Dockerfile checks for the new
engine sharding function and fails during build on that image. This workspace
started at `3d23551`; its dependency requirements also differ from the old
torch-2.11 image. Build and validate an engine image containing this patch with
the current project's dependency requirements first. Do not copy current Python
modules over the old environment or reuse its constraints unchanged.

Once that engine image has a new tag or immutable digest:

```bash
podman build --build-arg FT_BASE_IMAGE=freetoken:tp-engine \
  -t freetoken:sglang-tp-adapter -f Dockerfile.sglang .
```

Use the platform's internal registry and CA configuration for the actual tag.
No image is pushed or service deployed by this change.

Set the platform to SGLang, container/service/probe port 8000, the mounted model
directory, TP=2, metrics off, concurrency 2 and context length 16384. Leave
optional SGLang options unset initially. The adapter accepts `--tp`, `--tp-size`
and `--tensor-parallel-size`. It forwards the actual size and automatically uses
the text-only native NVFP4 expert path for TP>1.

Mappings retain FreeToken semantics:

| Platform option | FreeToken setting |
| --- | --- |
| `--model-path` | `--model` |
| `--context-length` | `--max-seq-len-override` |
| `--log-level` | `LOG_LEVEL` |
| `--disable-cuda-graph` | `--cuda-graph-max-bs 0` |
| Host, port, served name, concurrency | Passed through |
| `--enable-metrics` | Warning only; `/metrics` is absent |
| `--nnodes`, `--node-rank` | Only 1 node and rank 0 accepted |

Unknown arguments fail without printing their values. In particular,
`--mem-fraction-static` and `--cpu-offload-gb` have no approximate translation.
SGLang private endpoints and authentication flags are not implemented here.

Defaults can be set with `FT_TP_SIZE`, `FT_PLE_BACKEND`, `FT_COMM_BACKEND`,
`FT_DISTRIBUTED_TIMEOUT`, and the `FT_MAX_*`, `FT_KV_RESERVE_TOKENS`,
`FT_MEMORY_RATIO` variables from the supplied adapter. Explicit CLI arguments take
precedence. `FT_COMM_BACKEND=torch` and `FT_PLE_BACKEND=disk` are the new defaults;
set `FT_PLE_BACKEND=pinned` only with sufficient RAM. Preserve the base image's
writable cache mounts and non-root UID/GID 10001.

## Acceptance on the target cluster

1. Run the CPU and available GPU checks:
   `uv run pytest tests/models/qwen4_exp tests/layers/test_gdn.py tests/moe/test_nvfp4_backends.py -m "not slow"`.
2. Launch TP=1 with the exact checkpoint and record startup, peak RAM/VRAM, TTFT
   and generation rate. Repeat with TP=2 and otherwise identical settings.
   Collect logs from every rank, not just rank 0. Record hardware, driver,
   checkpoint ID, code revision and exact command with the results.
3. After loading completes, check both `/health` (`status=ok`) and `/v1/models`.
   An open HTTP port during loading is insufficient. Check the platform's real
   readiness/liveness probes and restart behavior.
4. Exercise greedy non-streaming and streaming requests, concurrent requests,
   a prompt beyond the QSA index budget, chunked prefill and prefix reuse.
   Compare answers and, when available, logits against TP=1. Floating-point
   reduction order can change tied greedy choices; coherent text alone does not
   establish numerical parity.
5. Repeat with `--disable-cuda-graph` if diagnosing graph/collective problems;
   validate the intended graph-enabled configuration before accepting it.

Inside the container:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
```

On the jump host, use the platform's CA trust and authentication. If its supplied
base URL already ends in `/v1`, append `/models` or `/chat/completions` directly.
Rollback means redeploying the existing TP=1 image and its original platform
settings; keep that image tag and service configuration available.

## Local verification

On the development host: Python 3.14.4, PyTorch 2.14.1+cpu, no NVIDIA GPU,
isolated dependencies under `/tmp`.

- Weight-reader and NVFP4 backend files: **46 passed, 22 skipped**.
- The acceptance-test command in step 1 above: **80 passed, 122 skipped,
  3 deselected, 1 failed**. The failure is
  `test_ple.py::test_track_snapshot_equals_a_prefill_stopped_at_the_boundary`;
  it also fails on an unmodified export of `3d23551` in this environment.
- Two real CPU/Gloo ranks: output projections, routed-expert reduction and
  padded vocabulary gather match their unsharded numerical references.
- Fault injection: wrong fused-head slicing and wrong down-scale rank each
  fail the new numerical assertions.
- Adapter subprocess/exec translation, actual FreeToken argument parsing,
  unknown-option value suppression, Python syntax and Docker RUN shell syntax
  passed. The image itself was not built or run here.

Related upstream work reviewed for this implementation:
[Qwen3.8 TP #385](https://github.com/FlashML-org/FreeToken/pull/385),
[Qwen3.5/3.6 TP #637](https://github.com/FlashML-org/FreeToken/pull/637), and
[the roadmap](https://github.com/FlashML-org/FreeToken/issues/79).
Their reported hardware measurements are not measurements of this patch.
