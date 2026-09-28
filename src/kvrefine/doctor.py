"""Capture machine capabilities without inventing inaccessible telemetry."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .records import JsonDict


def _run_command(args: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _package_versions() -> JsonDict:
    output: JsonDict = {}
    for name in ("numpy", "pytest", "mlx", "mlx-lm", "huggingface_hub", "zstandard"):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output


def collect_environment(*, smoke: bool) -> JsonDict:
    unavailable: JsonDict = {}
    result: JsonDict = {
        "schema_version": 1,
        "kind": "environment",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "architecture": platform.machine(),
        "packages": _package_versions(),
        "unavailable": unavailable,
        "application_budget_bytes": None,
    }
    observations = {
        "macos": ["sw_vers"],
        "cpu_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "ram_bytes": ["sysctl", "-n", "hw.memsize"],
        "swap": ["sysctl", "vm.swapusage"],
        "vm_stat": ["vm_stat"],
        "power_mode": ["pmset", "-g", "custom"],
    }
    for key, command in observations.items():
        code, stdout, stderr = _run_command(command)
        if code != 0 or not stdout:
            result[key] = None
            unavailable[key] = stderr or f"{command[0]} returned {code} without output"
            continue
        if key == "ram_bytes":
            try:
                result[key] = int(stdout)
            except ValueError:
                result[key] = None
                unavailable[key] = f"invalid hw.memsize result: {stdout}"
        else:
            result[key] = stdout
    try:
        usage = shutil.disk_usage(Path.cwd())
        result["disk_free_bytes"] = usage.free
        result["disk_total_bytes"] = usage.total
    except OSError as exc:
        result["disk_free_bytes"] = None
        result["disk_total_bytes"] = None
        unavailable["disk_free_bytes"] = str(exc)
    result["smoke_requested"] = smoke
    result["smoke"] = None
    result["native_abi"] = None
    if smoke:
        try:
            result["smoke"] = _attention_smoke()
            from .native import native_abi

            result["native_abi"] = native_abi()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            unavailable["smoke"] = str(exc)
    critical: list[str] = []
    if result["architecture"] != "arm64":
        critical.append("Python is not running on arm64")
    if sys.version_info[:2] != (3, 12):
        critical.append("Python 3.12 is required for this environment")
    if result["ram_bytes"] is None:
        critical.append("physical RAM capacity could not be measured")
    if result["disk_free_bytes"] is None:
        critical.append("free disk space could not be measured")
    for package in ("mlx", "mlx-lm"):
        if result["packages"][package] is None:
            critical.append(f"{package} is not installed")
    if smoke and result["smoke"] is None:
        critical.append("requested attention smoke has not run")
    result["critical_issues"] = critical
    return result


def _attention_smoke() -> JsonDict:
    import numpy as np
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.cache import QuantizedKVCache

    queries = mx.ones((1, 16, 1, 128), dtype=mx.bfloat16)
    keys = mx.ones((1, 8, 1, 128), dtype=mx.bfloat16)
    values = mx.array(np.arange(1024, dtype=np.float32).reshape(1, 8, 1, 128) / 1024, dtype=mx.bfloat16)
    raw = scaled_dot_product_attention(queries, keys, values, cache=None, scale=128 ** -0.5, mask=None)
    cache = QuantizedKVCache(group_size=64, bits=4)
    q_keys, q_values = cache.update_and_fetch(keys, values)
    approximate = scaled_dot_product_attention(
        queries, q_keys, q_values, cache=cache, scale=128 ** -0.5, mask=None
    )
    mx.eval(raw, approximate)
    if not np.isfinite(np.array(raw.astype(mx.float32))).all():
        raise RuntimeError("BF16 attention smoke produced nonfinite output")
    if not np.isfinite(np.array(approximate.astype(mx.float32))).all():
        raise RuntimeError("Q4 attention smoke produced nonfinite output")
    return {"bf16_shape": list(raw.shape), "q4_shape": list(approximate.shape), "native_capacity": cache.keys[0].shape[2]}
