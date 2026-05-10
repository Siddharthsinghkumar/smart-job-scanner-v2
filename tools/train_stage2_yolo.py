#!/usr/bin/env python3
"""Train Stage2 v3 YOLO detector with explicit reproducible config/report outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage2 YOLO v3 detector")
    parser.add_argument("--config", default="configs/stage2_yolo_v3.yaml", help="Training config YAML")
    parser.add_argument("--dataset-yaml", default=None, help="Override dataset YAML path")
    parser.add_argument("--output-report", default="run_state/stage2_v3_training_report.json", help="Training report JSON path")
    parser.add_argument("--weights-out", default="artifacts/stage2_yolo_v3/best.pt", help="Stable best-checkpoint output path")
    parser.add_argument("--last-weights-out", default="artifacts/stage2_yolo_v3/last.pt", help="Stable last-checkpoint output path")
    parser.add_argument(
        "--device",
        default=None,
        help="Resolved execution device override (e.g. cpu, 0, cuda:0)",
    )
    parser.add_argument(
        "--requested-device",
        default=None,
        help="Original requested device string for reporting (e.g. auto, cpu, 0, cuda:0)",
    )
    parser.add_argument(
        "--device-preflight-json",
        default=None,
        help="Optional preflight JSON path captured before training invocation",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required for training config parsing") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid config format: {path}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.config).resolve()
    report_path = (PROJECT_ROOT / args.output_report).resolve()
    weights_out = (PROJECT_ROOT / args.weights_out).resolve()
    last_weights_out = (PROJECT_ROOT / args.last_weights_out).resolve()

    if not cfg_path.is_file():
        raise SystemExit(f"Config file not found: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    dataset_yaml = Path(args.dataset_yaml).resolve() if args.dataset_yaml else (PROJECT_ROOT / cfg.get("dataset", {}).get("yaml_path", "")).resolve()
    if not dataset_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {dataset_yaml}")

    start_utc = datetime.now(timezone.utc).isoformat()
    command_preview = [
        sys.executable,
        "tools/train_stage2_yolo.py",
        "--config",
        str(cfg_path),
        "--dataset-yaml",
        str(dataset_yaml),
        "--output-report",
        str(report_path),
    ]
    if args.device:
        command_preview.extend(["--device", str(args.device)])
    if args.requested_device:
        command_preview.extend(["--requested-device", str(args.requested_device)])
    if args.device_preflight_json:
        command_preview.extend(["--device-preflight-json", str(args.device_preflight_json)])

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        report = {
            "status": "failed",
            "error": f"ultralytics_import_failed: {exc}",
            "started_at_utc": start_utc,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(cfg_path),
            "dataset_yaml": str(dataset_yaml),
            "command": command_preview,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    model_name = str(train_cfg.get("model", "yolov8n.pt"))
    requested_device = str(args.requested_device or train_cfg.get("device", "cpu"))
    resolved_device = str(args.device or train_cfg.get("device", "cpu"))
    preflight_path = Path(args.device_preflight_json).resolve() if args.device_preflight_json else None
    preflight_payload = _load_json_if_exists(preflight_path)

    # Correctness-first baseline run; speed is explicitly not optimized here.
    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": int(train_cfg.get("epochs", 80)),
        "imgsz": int(train_cfg.get("imgsz", 1280)),
        "batch": int(train_cfg.get("batch", 4)),
        "device": resolved_device,
        "workers": int(train_cfg.get("workers", 0)),
        "seed": int(train_cfg.get("seed", 20260331)),
        "project": str((PROJECT_ROOT / str(train_cfg.get("project", "artifacts/stage2_yolo_v3/runs"))).resolve()),
        "name": str(train_cfg.get("name", "baseline")),
        "pretrained": bool(train_cfg.get("pretrained", True)),
        "cache": bool(train_cfg.get("cache", False)),
        "deterministic": bool(train_cfg.get("deterministic", True)),
        "patience": int(train_cfg.get("patience", 25)),
        "close_mosaic": int(train_cfg.get("close_mosaic", 10)),
        "optimizer": str(train_cfg.get("optimizer", "auto")),
        "lr0": float(train_cfg.get("lr0", 0.01)),
        "verbose": bool(train_cfg.get("verbose", True)),
    }

    model = YOLO(model_name)
    results = model.train(**train_kwargs)

    save_dir = Path(str(getattr(results, "save_dir", ""))).resolve()
    weights_dir = save_dir / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"

    weights_out.parent.mkdir(parents=True, exist_ok=True)
    last_weights_out.parent.mkdir(parents=True, exist_ok=True)

    copied_best = None
    copied_last = None
    if best_ckpt.is_file():
        shutil.copy2(best_ckpt, weights_out)
        copied_best = str(weights_out)
    if last_ckpt.is_file():
        shutil.copy2(last_ckpt, last_weights_out)
        copied_last = str(last_weights_out)

    metrics = getattr(results, "results_dict", {})

    # Keep raw ultralytics CLI-proximal info for reproducibility.
    report = {
        "status": "ok",
        "started_at_utc": start_utc,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "dataset_yaml": str(dataset_yaml),
        "command": command_preview,
        "train_kwargs": _json_safe(train_kwargs),
        "device": {
            "requested_device": requested_device,
            "resolved_device": resolved_device,
            "preflight_json_path": str(preflight_path) if preflight_path else None,
            "preflight": _json_safe(preflight_payload) if preflight_payload else None,
            "cuda_available": (
                bool(preflight_payload.get("torch", {}).get("cuda_is_available", False))
                if preflight_payload
                else None
            ),
            "execution_mode": (
                str(preflight_payload.get("execution_mode"))
                if preflight_payload and preflight_payload.get("execution_mode") is not None
                else None
            ),
            "fallback_used": (
                bool(preflight_payload.get("fallback_used", False))
                if preflight_payload
                else None
            ),
        },
        "model_init": model_name,
        "artifacts": {
            "save_dir": str(save_dir),
            "weights_dir": str(weights_dir),
            "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
            "last_checkpoint": str(last_ckpt) if last_ckpt.exists() else None,
            "stable_best_checkpoint": copied_best,
            "stable_last_checkpoint": copied_last,
        },
        "validation_metrics": _json_safe(metrics),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Mirror training invocation for shell reproducibility.
    run_cmd_txt = PROJECT_ROOT / "run_state" / "stage2_v3_training_command.txt"
    run_cmd_txt.parent.mkdir(parents=True, exist_ok=True)
    run_cmd_txt.write_text(" ".join(command_preview) + "\n", encoding="utf-8")

    print(f"train save_dir: {save_dir}")
    print(f"best checkpoint: {best_ckpt if best_ckpt.exists() else 'missing'}")
    print(f"stable best checkpoint: {copied_best}")
    print(f"training report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
