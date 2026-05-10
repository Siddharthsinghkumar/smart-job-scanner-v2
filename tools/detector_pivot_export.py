#!/usr/bin/env python3
"""Export detector-pivot YOLO checkpoint for CPU-friendly deployment formats."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.detector_device import resolve_device_with_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export detector-pivot checkpoint to ONNX/OpenVINO")
    parser.add_argument(
        "--weights",
        default="artifacts/detector_pivot_yolo_v1/best.pt",
        help="Trained YOLO weights path",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Export image size")
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device request: auto | cpu | 0 | cuda:0 (or other raw backend string)",
    )
    parser.add_argument(
        "--device-preflight-report",
        default="run_state/detector_pivot_device_preflight.json",
        help="Device preflight report JSON path",
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "openvino"],
        choices=["onnx", "openvino"],
        help="Export formats",
    )
    parser.add_argument(
        "--output-report",
        default="run_state/detector_pivot_export_report.json",
        help="Export report JSON path",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def main() -> int:
    args = parse_args()
    weights = Path(args.weights).resolve()
    report_path = Path(args.output_report).resolve()
    preflight_report_path = (
        Path(args.device_preflight_report).resolve()
        if Path(args.device_preflight_report).is_absolute()
        else (PROJECT_ROOT / args.device_preflight_report).resolve()
    )
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")
    preflight = resolve_device_with_preflight(
        requested_device=args.device,
        context="detector_pivot_export",
        preflight_report_path=preflight_report_path,
    )
    selected_device = str(preflight.get("selected_device", "cpu"))

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": f"ultralytics_import_failed: {exc}",
            "weights": str(weights),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    model = YOLO(str(weights))
    outputs: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    for fmt in args.formats:
        try:
            kwargs = {
                "format": fmt,
                "imgsz": int(args.imgsz),
                "device": selected_device,
            }
            if fmt == "onnx":
                kwargs["opset"] = int(args.opset)
                kwargs["dynamic"] = False
                kwargs["simplify"] = True
            exported = model.export(**kwargs)
            outputs[fmt] = {
                "status": "ok",
                "export_result": _json_safe(exported),
                "kwargs": _json_safe(kwargs),
            }
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            outputs[fmt] = {"status": "failed"}
            errors.append({"format": fmt, "error": str(exc)})

    payload = {
        "status": "ok" if not errors else "partial_failure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "requested_device": str(args.device),
        "selected_device": selected_device,
        "execution_mode": str(preflight.get("execution_mode", "cpu")),
        "device_preflight_report": str(preflight_report_path),
        "device_preflight": preflight,
        "formats_requested": args.formats,
        "outputs": outputs,
        "errors": errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
