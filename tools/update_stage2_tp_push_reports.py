#!/usr/bin/env python3
"""Update TP-push iteration report and persistent best-so-far record."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_STATE = PROJECT_ROOT / "run_state"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_tp(metrics: dict[str, Any]) -> int:
    return int(metrics.get("true_positives", 0) or 0)


def _candidate_key(iter_payload: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(iter_payload["detector"]["tp"]),
        int(iter_payload["passthrough"]["tp"]),
        int(iter_payload["normal_refined"]["tp"]),
    )


def _build_worst_pages(page_rows: list[dict[str, Any]], top_n: int = 8) -> dict[str, Any]:
    rows = list(page_rows)
    hardest_detector = sorted(
        rows,
        key=lambda r: (
            int(r.get("labeled_box_count", 0)) - int(r.get("detector_tp", 0)),
            int(r.get("labeled_box_count", 0)),
        ),
        reverse=True,
    )[:top_n]
    zero_hit_pages = [
        r
        for r in rows
        if int(r.get("labeled_box_count", 0)) > 0
        and int(r.get("detector_tp", 0)) == 0
        and int(r.get("passthrough_tp", 0)) == 0
        and int(r.get("normal_refined_tp", 0)) == 0
    ]
    detector_vs_normal_gap = sorted(
        rows,
        key=lambda r: int(r.get("detector_tp", 0)) - int(r.get("normal_refined_tp", 0)),
        reverse=True,
    )[:top_n]
    return {
        "hardest_detector_pages": hardest_detector,
        "zero_hit_pages": zero_hit_pages,
        "detector_improved_but_normal_not": detector_vs_normal_gap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update TP push iteration and best-so-far reports")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--detector-version", default="v4")
    parser.add_argument("--stage3-mode", default="normal")
    parser.add_argument("--params-file", default="configs/detection_params_v4.json")
    parser.add_argument("--comparison", default="run_state/stage2_recall_push_comparison.json")
    parser.add_argument("--page-report", default="run_state/stage2_recall_push_page_report.json")
    parser.add_argument("--best-file", default="run_state/stage2_tp_push_best_so_far.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison_path = (PROJECT_ROOT / args.comparison).resolve()
    page_report_path = (PROJECT_ROOT / args.page_report).resolve()
    params_path = (PROJECT_ROOT / args.params_file).resolve()
    best_path = (PROJECT_ROOT / args.best_file).resolve()

    comparison = _load_json(comparison_path)
    page_report = _load_json(page_report_path)
    params = _load_json(params_path)

    total = int(comparison.get("total_labeled_ads", 0) or 0)
    cand = comparison.get("candidate", {})
    detector = cand.get("detector", {})
    passthrough = cand.get("passthrough", {})
    normal = cand.get("normal_refined", {})

    iter_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": int(args.iteration),
        "detector_version": args.detector_version,
        "stage3_mode": args.stage3_mode,
        "params_file": str(params_path),
        "exact_params_used": params,
        "total_labeled_ads": total,
        "detector": {
            "tp": _safe_tp(detector),
            "fp": int(detector.get("false_positives", 0) or 0),
            "fn": int(detector.get("missed_detections", 0) or 0),
            "recall": float(detector.get("recall", 0.0) or 0.0),
            "precision": float(detector.get("precision", 0.0) or 0.0),
            "total_detections": int(detector.get("total_detected_ads", 0) or 0),
            "tp_over_total": f"{_safe_tp(detector)}/{total}",
        },
        "passthrough": {
            "tp": _safe_tp(passthrough),
            "fp": int(passthrough.get("false_positives", 0) or 0),
            "fn": int(passthrough.get("missed_detections", 0) or 0),
            "recall": float(passthrough.get("recall", 0.0) or 0.0),
            "precision": float(passthrough.get("precision", 0.0) or 0.0),
            "total_detections": int(passthrough.get("total_detected_ads", 0) or 0),
            "tp_over_total": f"{_safe_tp(passthrough)}/{total}",
        },
        "normal_refined": {
            "tp": _safe_tp(normal),
            "fp": int(normal.get("false_positives", 0) or 0),
            "fn": int(normal.get("missed_detections", 0) or 0),
            "recall": float(normal.get("recall", 0.0) or 0.0),
            "precision": float(normal.get("precision", 0.0) or 0.0),
            "total_detections": int(normal.get("total_detected_ads", 0) or 0),
            "tp_over_total": f"{_safe_tp(normal)}/{total}",
        },
        "worst_pages": _build_worst_pages(page_report.get("rows", [])),
        "source_artifacts": {
            "comparison": str(comparison_path),
            "page_report": str(page_report_path),
            "overlays_dir": str((PROJECT_ROOT / "run_state" / "stage2_recall_push_overlays").resolve()),
        },
    }

    best_doc: dict[str, Any]
    if best_path.exists():
        best_doc = _load_json(best_path)
    else:
        best_doc = {
            "generated_at_utc": iter_payload["generated_at_utc"],
            "best": None,
            "history": [],
        }

    previous_best = best_doc.get("best")
    becomes_best = False
    if not isinstance(previous_best, dict):
        becomes_best = True
    else:
        prev_key = _candidate_key(previous_best)
        curr_key = _candidate_key(iter_payload)
        becomes_best = curr_key > prev_key

    iter_payload["is_new_best_so_far"] = bool(becomes_best)

    iter_path = RUN_STATE / f"stage2_tp_push_iteration_{int(args.iteration)}.json"
    iter_path.write_text(json.dumps(iter_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if becomes_best:
        best_doc["best"] = iter_payload
    best_doc.setdefault("history", []).append(
        {
            "iteration": int(args.iteration),
            "detector_tp": int(iter_payload["detector"]["tp"]),
            "passthrough_tp": int(iter_payload["passthrough"]["tp"]),
            "normal_refined_tp": int(iter_payload["normal_refined"]["tp"]),
            "is_new_best_so_far": bool(becomes_best),
            "timestamp_utc": iter_payload["generated_at_utc"],
            "iteration_report": str(iter_path.resolve()),
        }
    )
    best_doc["last_updated_utc"] = iter_payload["generated_at_utc"]
    best_doc["best_so_far"] = best_doc.get("best")
    best_path.write_text(json.dumps(best_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"iteration_report={iter_path}")
    print(f"best_file={best_path}")
    print(f"is_new_best_so_far={int(bool(becomes_best))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
