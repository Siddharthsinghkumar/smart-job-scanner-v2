"""Device preflight and resolution helpers for detector pivot scripts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str(value: Any) -> str:
    if value is None:
        return "auto"
    text = str(value).strip()
    return text if text else "auto"


def _is_explicit_gpu_request(device: str) -> bool:
    d = device.lower().strip()
    if d in {"0", "cuda", "gpu"}:
        return True
    if d.isdigit():
        return True
    return d.startswith("cuda:")


def _normalize_device_for_ultralytics(device: str) -> str:
    d = device.strip().lower()
    if d in {"auto", ""}:
        return "auto"
    if d == "gpu" or d == "cuda":
        return "0"
    return device.strip()


def _collect_torch_cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": None,
        "cuda_is_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
        "torch_import_error": None,
    }
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime/env dependent
        info["torch_import_error"] = str(exc)
        return info

    info["torch_version"] = str(getattr(torch, "__version__", "unknown"))
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - runtime/env dependent
        info["torch_cuda_probe_error"] = str(exc)
        return info

    info["cuda_is_available"] = cuda_available
    info["cuda_device_count"] = device_count
    if cuda_available and device_count > 0:
        names: list[str] = []
        for idx in range(device_count):
            try:
                names.append(str(torch.cuda.get_device_name(idx)))
            except Exception:
                names.append(f"cuda:{idx}")
        info["cuda_device_names"] = names
    return info


def _validate_requested_gpu_index(device: str, device_count: int) -> str | None:
    d = device.strip().lower()
    if d.isdigit():
        idx = int(d)
        if idx >= device_count:
            return f"GPU index {idx} requested but only {device_count} CUDA device(s) detected."
    if d.startswith("cuda:"):
        suffix = d.split(":", 1)[1].strip()
        if suffix.isdigit():
            idx = int(suffix)
            if idx >= device_count:
                return f"Device {device!r} requested but only {device_count} CUDA device(s) detected."
    return None


def resolve_device_with_preflight(
    *,
    requested_device: str | None,
    context: str,
    preflight_report_path: str | Path,
) -> dict[str, Any]:
    """Resolve execution device and persist a reusable preflight report.

    Rules:
    - requested=auto -> GPU 0 when CUDA is available, else CPU fallback.
    - requested=cpu -> CPU always.
    - requested explicit GPU (e.g. 0/cuda/cuda:0) -> fail-fast if CUDA unavailable.
    """

    requested_raw = _as_str(requested_device)
    requested_norm = _normalize_device_for_ultralytics(requested_raw)
    explicit_gpu_requested = _is_explicit_gpu_request(requested_norm)
    torch_info = _collect_torch_cuda_info()
    cuda_available = bool(torch_info.get("cuda_is_available", False))
    device_count = int(torch_info.get("cuda_device_count", 0) or 0)

    selected_device = "cpu"
    execution_mode = "cpu"
    fallback_used = False
    fallback_reason = None
    status = "ok"
    error = None

    if requested_norm.lower() == "auto":
        if cuda_available and device_count > 0:
            selected_device = "0"
            execution_mode = "gpu"
        else:
            selected_device = "cpu"
            execution_mode = "cpu"
            fallback_used = True
            fallback_reason = "auto_selected_cpu_because_cuda_unavailable"
    elif requested_norm.lower() == "cpu":
        selected_device = "cpu"
        execution_mode = "cpu"
    elif explicit_gpu_requested:
        if not cuda_available or device_count <= 0:
            status = "failed"
            error = (
                f"GPU device {requested_raw!r} was explicitly requested, "
                "but torch.cuda.is_available() is False / no CUDA devices detected."
            )
        else:
            index_error = _validate_requested_gpu_index(requested_norm, device_count)
            if index_error:
                status = "failed"
                error = index_error
            else:
                selected_device = requested_norm
                execution_mode = "gpu"
    else:
        # Raw passthrough for non-standard backends supported by underlying tool.
        selected_device = requested_norm
        execution_mode = "cpu" if requested_norm.lower() == "cpu" else "custom"

    payload = {
        "generated_at_utc": _utc_now(),
        "context": str(context),
        "status": status,
        "requested_device": requested_raw,
        "selected_device": selected_device,
        "execution_mode": execution_mode,
        "explicit_gpu_requested": explicit_gpu_requested,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "error": error,
        "torch": torch_info,
    }

    report_path = Path(preflight_report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if status != "ok":
        raise RuntimeError(error or "device preflight failed")
    return payload
