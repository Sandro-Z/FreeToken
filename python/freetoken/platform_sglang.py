"""Translate the platform's SGLang launch command into a FreeToken process."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

FT_EXECUTABLE = "/opt/venv/bin/ft"
ADAPTER_ROOT = "/opt/freetoken-platform"


def positive_int(value: str) -> int:
    try:
        result = int(value)
        if result > 0:
            return result
    except (ValueError, TypeError):
        pass
    raise argparse.ArgumentTypeError("must be a positive integer")


def positive_float(value: str) -> float:
    try:
        result = float(value)
        if math.isfinite(result) and result > 0:
            return result
    except (ValueError, TypeError):
        pass
    raise argparse.ArgumentTypeError("must be a finite positive number")


def build_launch(argv: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-path", "--model", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME"))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=positive_int, default=os.environ.get("PORT", "8000"))
    parser.add_argument("--tensor-parallel-size", "--tp-size", "--tp", type=positive_int,
                        default=os.environ.get("FT_TP_SIZE", "1"))
    parser.add_argument("--nnodes", type=positive_int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--gpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-metrics", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--text-model-only", action="store_true")
    parser.add_argument("--log-level", type=str.lower,
                        choices=("debug", "info", "warning", "error", "critical"),
                        default=os.environ.get("LOG_LEVEL", "info").lower())
    parser.add_argument("--context-length", "--max-seq-len-override", dest="max_seq_len",
                        type=positive_int, default=os.environ.get("FT_MAX_SEQ_LEN", "16384"))
    for flag, env, default in (
        ("max-running-requests", "FT_MAX_RUNNING_REQUESTS", "2"),
        ("max-output-tokens", "FT_MAX_OUTPUT_TOKENS", "2048"),
        ("max-prefill-length", "FT_MAX_PREFILL_LENGTH", "4096"),
        ("kv-reserve-tokens", "FT_KV_RESERVE_TOKENS", "32768"),
    ):
        parser.add_argument("--" + flag, type=positive_int, default=os.environ.get(env, default))
    parser.add_argument("--memory-ratio", type=positive_float,
                        default=os.environ.get("FT_MEMORY_RATIO", "0.90"))
    parser.add_argument("--distributed-timeout", type=positive_float,
                        default=os.environ.get("FT_DISTRIBUTED_TIMEOUT", "1800"))
    parser.add_argument("--ple-backend", choices=("disk", "pinned"),
                        default=os.environ.get("FT_PLE_BACKEND", "disk"))
    parser.add_argument("--comm-backend", choices=("torch", "pynccl"),
                        default=os.environ.get("FT_COMM_BACKEND", "torch"))
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        options = sorted({item.partition("=")[0] for item in unknown if item.startswith("--")})
        parser.error("unsupported platform arguments: " + (", ".join(options) or "unexpected positional values"))
    if args.nnodes != 1 or args.node_rank != 0:
        parser.error("this adapter supports one node per container; multi-node launch is not implemented")
    if args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.memory_ratio >= 1:
        parser.error("--memory-ratio must be between 0 and 1")
    if args.comm_backend not in ("torch", "pynccl") or args.ple_backend not in ("disk", "pinned"):
        parser.error("invalid FT_COMM_BACKEND or FT_PLE_BACKEND")
    if args.log_level not in ("debug", "info", "warning", "error", "critical"):
        parser.error("invalid LOG_LEVEL")
    if not args.model_path or not (Path(args.model_path) / "config.json").is_file():
        parser.error("--model-path or MODEL_PATH must name a mounted directory containing config.json")
    if args.tensor_parallel_size not in (1, 2, 4, 8):
        parser.error("the Qwen3.8 platform profile supports TP=1, 2, 4 or 8")
    if args.tensor_parallel_size > 1:
        try:
            with (Path(args.model_path) / "config.json").open() as file:
                config = json.load(file)
        except (OSError, ValueError):
            parser.error("model config.json cannot be read as JSON")
        if not isinstance(config, dict) or config.get("model_type") != "qwen4_exp":
            parser.error("this multi-GPU platform profile requires a qwen4_exp checkpoint")
    if args.enable_metrics:
        print("[freetoken-platform] WARNING: /metrics is not implemented; disable platform metrics.",
              file=sys.stderr, flush=True)
    if args.tensor_parallel_size > 1:
        print("[freetoken-platform] Qwen3.8 TP serves text only; use NVFP4 experts with whole 16-wide scale blocks.",
              file=sys.stderr, flush=True)
    served_name = args.served_model_name or Path(args.model_path).name
    env = os.environ.copy()
    env.update(LOG_LEVEL=args.log_level.upper(), PORT=str(args.port),
               MODEL_PATH=args.model_path, SERVED_MODEL_NAME=served_name)
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != ADAPTER_ROOT]
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths)
    else:
        env.pop("PYTHONPATH", None)
    command = [FT_EXECUTABLE, "serve", "--model", args.model_path, "--served-model-name", served_name,
               "--host", args.host, "--port", str(args.port),
               "--tensor-parallel-size", str(args.tensor_parallel_size),
               "--moe-strategy", "offload", "--ple-backend", args.ple_backend, "--expert-load", "serial",
               "--max-running-requests", str(args.max_running_requests),
               "--max-seq-len-override", str(args.max_seq_len), "--max-output-tokens", str(args.max_output_tokens),
               "--max-prefill-length", str(args.max_prefill_length), "--kv-reserve-tokens", str(args.kv_reserve_tokens),
               "--memory-ratio", str(args.memory_ratio), "--distributed-timeout", str(args.distributed_timeout)]
    if args.comm_backend == "torch":
        command.append("--disable-pynccl")
    if args.disable_cuda_graph:
        command.extend(["--cuda-graph-max-bs", "0"])
    if args.text_model_only or args.tensor_parallel_size > 1:
        command.append("--text-model-only")
    if args.tensor_parallel_size > 1:
        command.extend(["--quant-backend", "moe.nvfp4=triton"])
    if args.gpu:
        command.extend(["--gpu", args.gpu])
    return command, env


def main(argv: list[str] | None = None) -> None:
    command, env = build_launch(argv)
    if not os.access(FT_EXECUTABLE, os.X_OK):
        raise SystemExit("FreeToken executable is missing or inaccessible")
    try:
        for key in ("HOME", "XDG_CACHE_HOME", "HF_HOME", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
            if env.get(key):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
    except OSError:
        raise SystemExit("FreeToken cache directory is not writable") from None
    print("[freetoken-platform] starting FreeToken", flush=True)
    os.execve(FT_EXECUTABLE, command, env)


if __name__ == "__main__":
    main()
