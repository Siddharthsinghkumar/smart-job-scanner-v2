#!/usr/bin/env python3
"""Streaming Stage1->Stage2->Stage3 runner with adaptive guardrails."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

try:
    import psutil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime env dependent
    psutil = None

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime env dependent
    torch = None

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.pipeline_config import get_path, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS_PATH = PROJECT_ROOT / "run_state" / "pipeline_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming stage01-stage03 runner with guardrails")
    parser.add_argument("--config", default="configs/pipeline_paths.json", help="Path to pipeline paths config")
    parser.add_argument("--image-queue-max", type=int, default=50, help="Max queued page images")
    parser.add_argument("--block-queue-max", type=int, default=100, help="Max queued stage2 page outputs")
    parser.add_argument(
        "--metadata-queue-max",
        type=int,
        default=400,
        help="Max queued metadata updates for detections writer",
    )
    parser.add_argument("--sample-interval", type=float, default=2.0, help="Resource sampling interval in seconds")
    parser.add_argument("--stats-path", default=str(DEFAULT_STATS_PATH), help="Pipeline runtime stats output path")
    parser.add_argument(
        "--stage2-debug-sample-rate",
        type=float,
        default=0.0,
        help="Probability of debug image writes for stage2 pages in streaming mode",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=600,
        help="Max recent samples to retain in pipeline_stats.json",
    )
    return parser.parse_args()


def _queue_size_safe(q) -> int:
    try:
        return int(q.qsize())
    except Exception:
        return -1


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_base_workers(cpu_total: int) -> dict[str, int]:
    return {"s1": 1, "s2": max(2, cpu_total // 3), "s3": max(1, cpu_total // 6)}


def _effective_cpu_budget(cpu_total: int) -> int:
    return max(1, int(math.floor(0.8 * cpu_total)))


def _gpu_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _apply_gpu_downscale(limits: dict[str, int], gpu_available: bool) -> dict[str, int]:
    if not gpu_available:
        return dict(limits)
    scaled = {
        "s1": max(1, int(math.floor(limits["s1"] * 0.7))),
        "s2": max(1, int(math.floor(limits["s2"] * 0.7))),
        "s3": max(1, int(math.floor(limits["s3"] * 0.7))),
    }
    return scaled


def _clamp_limits_to_budget(limits: dict[str, int], budget: int, cpu_total: int) -> dict[str, int]:
    out = dict(limits)
    min_s1 = 1
    min_s2 = 1
    min_s3 = 1
    while (out["s1"] + out["s2"] + out["s3"]) > budget:
        if out["s2"] > min_s2:
            out["s2"] -= 1
        elif out["s3"] > min_s3:
            out["s3"] -= 1
        elif cpu_total == 1 and out["s1"] > min_s1:
            out["s1"] -= 1
        else:
            break
    return out


def _set_limits(limit_values: dict[str, Any], limits: dict[str, int]) -> None:
    for stage in ("s1", "s2", "s3"):
        limit_values[stage].value = int(max(1, limits[stage]))


def _get_limits(limit_values: dict[str, Any]) -> dict[str, int]:
    return {stage: int(limit_values[stage].value) for stage in ("s1", "s2", "s3")}


def _safe_put(queue_obj, item, timeout_seconds: float = 30.0) -> bool:
    deadline = time.time() + max(0.5, float(timeout_seconds))
    while time.time() < deadline:
        try:
            queue_obj.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
        except Exception:
            return False
    return False


def _record_stats(stats_path: Path, payload: dict[str, Any], max_samples: int) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_path.exists():
        try:
            existing = json.loads(stats_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
    else:
        existing = {}

    samples = existing.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    samples.append(payload)
    if len(samples) > max_samples:
        samples = samples[-max_samples:]

    existing["samples"] = samples
    existing["last_updated"] = _ts()
    stats_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def _stage_is_active(worker_idx: int, limit_value) -> bool:
    return worker_idx < int(limit_value.value)


def _stage1_worker(
    worker_idx: int,
    pdf_queue,
    stage1_input_closed,
    image_queue,
    images_output_dir: str,
    stage1_limit,
    stage1_pause_event,
    counters,
):
    from src.pipeline.stage01_pdf_to_images import process_pdf_streaming

    while True:
        if not _stage_is_active(worker_idx, stage1_limit):
            time.sleep(0.1)
            continue
        if stage1_pause_event.is_set():
            time.sleep(0.1)
            continue

        try:
            pdf_path = pdf_queue.get(timeout=0.3)
        except queue.Empty:
            if stage1_input_closed.is_set():
                break
            continue

        pdf_path = str(pdf_path)

        def _on_page_done(page_event: dict[str, Any]) -> None:
            image_queue.put(page_event)
            with counters["stage1_pages"].get_lock():
                counters["stage1_pages"].value += 1

        try:
            process_pdf_streaming(
                pdf_path=Path(pdf_path),
                images_output_dir=images_output_dir,
                on_page_done=_on_page_done,
                move_processed=True,
            )
        except Exception:
            with counters["stage1_failures"].get_lock():
                counters["stage1_failures"].value += 1


def _stage2_worker(
    worker_idx: int,
    image_queue,
    stage2_input_closed,
    block_queue,
    metadata_queue,
    blocks_output_dir: str,
    stage2_limit,
    counters,
    debug_sample_rate: float,
):
    from src.pipeline.stage02_block_detection import (
        _export_detector_env_once,
        _load_detection_params,
        detect_page_blocks,
    )

    detector_params = _load_detection_params()
    _export_detector_env_once(detector_params)

    sample_rate = max(0.0, min(1.0, float(debug_sample_rate)))

    while True:
        if not _stage_is_active(worker_idx, stage2_limit):
            time.sleep(0.1)
            continue

        try:
            page_event = image_queue.get(timeout=0.3)
        except queue.Empty:
            if stage2_input_closed.is_set():
                break
            continue

        if page_event is None:
            break

        page_path = str(page_event.get("page_path", ""))
        if not page_path:
            continue

        debug = False
        if sample_rate > 0.0:
            # deterministic sampling based on page name
            page_name = str(page_event.get("page_name", ""))
            deterministic = (sum(ord(ch) for ch in page_name) % 1000) / 1000.0
            debug = deterministic < sample_rate

        result = detect_page_blocks(page_path, blocks_output_dir, debug=debug)

        if result["status"] == "processed":
            metadata_queue.put(
                {
                    "stage": "detector",
                    "page_name": result["page_name"],
                    "detections": result["detections"],
                }
            )
            block_queue.put(
                {
                    "pdf_folder": result["pdf_folder"],
                    "page_name": result["page_name"],
                    "block_paths": result["block_paths"],
                    "detector_count": len(result["detections"]),
                }
            )
            with counters["stage2_pages"].get_lock():
                counters["stage2_pages"].value += 1
            with counters["stage2_blocks"].get_lock():
                counters["stage2_blocks"].value += len(result["detections"])
        elif result["status"] == "skipped":
            with counters["stage2_skipped"].get_lock():
                counters["stage2_skipped"].value += 1
        else:
            with counters["stage2_failures"].get_lock():
                counters["stage2_failures"].value += 1


def _stage3_worker(
    worker_idx: int,
    block_queue,
    stage3_input_closed,
    metadata_queue,
    blocks_output_dir: str,
    refined_output_dir: str,
    detections_output_dir: str,
    stage3_limit,
    counters,
):
    from src.pipeline.stage03_block_refiner import refine_page_blocks

    while True:
        if not _stage_is_active(worker_idx, stage3_limit):
            time.sleep(0.1)
            continue

        try:
            block_event = block_queue.get(timeout=0.3)
        except queue.Empty:
            if stage3_input_closed.is_set():
                break
            continue

        if block_event is None:
            break

        pdf_folder = str(block_event.get("pdf_folder", ""))
        page_name = str(block_event.get("page_name", ""))
        if not pdf_folder or not page_name:
            continue

        expected_detector_count = int(block_event.get("detector_count", 0) or 0)
        if expected_detector_count > 0:
            _wait_for_detector_metadata(
                page_name=page_name,
                detections_output_dir=detections_output_dir,
                expected_count=expected_detector_count,
                timeout_seconds=10.0,
            )

        result = refine_page_blocks(
            pdf_folder=pdf_folder,
            page_name=page_name,
            input_base=blocks_output_dir,
            output_base=refined_output_dir,
            detections_output=detections_output_dir,
            write_metadata=False,
            enable_debug_sampling=False,
        )
        metadata_queue.put(
            {
                "stage": "refined",
                "page_name": page_name,
                "detections": result.get("refined_metadata", []),
            }
        )
        with counters["stage3_pages"].get_lock():
            counters["stage3_pages"].value += 1
        with counters["stage3_refined"].get_lock():
            counters["stage3_refined"].value += int(result.get("saved", 0))


def _wait_for_detector_metadata(
    page_name: str,
    detections_output_dir: str,
    expected_count: int,
    timeout_seconds: float = 10.0,
) -> None:
    detections_path = Path(detections_output_dir) / f"{page_name}.json"
    deadline = time.time() + max(0.2, float(timeout_seconds))
    while time.time() < deadline:
        if detections_path.exists():
            try:
                payload = json.loads(detections_path.read_text(encoding="utf-8"))
                rows = payload.get("detections", []) if isinstance(payload, dict) else []
                detector_rows = [
                    row for row in rows
                    if isinstance(row, dict) and row.get("stage") == "detector"
                ]
                if len(detector_rows) >= int(expected_count):
                    return
            except Exception:
                pass
        time.sleep(0.05)


def _metadata_writer_worker(metadata_queue, metadata_input_closed, detections_output_dir: str, counters):
    from src.pipeline.stage02_block_detection import _write_detector_metadata
    from src.pipeline.stage03_block_refiner import _upsert_refined_metadata

    detections_dir = Path(detections_output_dir)
    detections_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            event = metadata_queue.get(timeout=0.3)
        except queue.Empty:
            if metadata_input_closed.is_set():
                break
            continue

        if event is None:
            break

        stage = event.get("stage")
        page_name = str(event.get("page_name", ""))
        detections = event.get("detections", [])
        if not page_name:
            continue

        try:
            if stage == "detector":
                _write_detector_metadata(page_name, detections, detections_dir)
            elif stage == "refined":
                _upsert_refined_metadata(page_name, detections, detections_dir)
            with counters["metadata_writes"].get_lock():
                counters["metadata_writes"].value += 1
        except Exception:
            with counters["metadata_failures"].get_lock():
                counters["metadata_failures"].value += 1


def _controller_loop(
    stop_event: threading.Event,
    teardown_mode: threading.Event,
    limit_values,
    base_limits: dict[str, int],
    budget: int,
    queues: dict[str, Any],
    counters,
    stage1_pause_event,
    stats_path: Path,
    sample_interval: float,
    max_samples: int,
    cpu_total: int,
):
    if psutil is not None:
        psutil.cpu_percent(interval=None)
        prev_disk = psutil.disk_io_counters()
    else:
        prev_disk = None
    prev_time = time.time()
    prev_stage3_pages = int(counters["stage3_pages"].value)
    last_progress_print = 0.0

    while not stop_event.is_set():
        time.sleep(max(0.5, sample_interval))

        now = time.time()
        dt = max(0.001, now - prev_time)

        if psutil is not None:
            cpu_pct = float(psutil.cpu_percent(interval=None))
            mem_pct = float(psutil.virtual_memory().percent)
        else:
            cpu_pct = 0.0
            mem_pct = 0.0
        try:
            load_1m = float(os.getloadavg()[0])
        except Exception:
            load_1m = 0.0

        disk_now = psutil.disk_io_counters() if psutil is not None else None
        read_pressure = False
        write_pressure = False
        read_time_delta = 0.0
        write_time_delta = 0.0
        if prev_disk and disk_now:
            read_time_delta = float(max(0, (disk_now.read_time or 0) - (prev_disk.read_time or 0)))
            write_time_delta = float(max(0, (disk_now.write_time or 0) - (prev_disk.write_time or 0)))
            # ms/s proxy for I/O pressure
            read_pressure = (read_time_delta / dt) > 450.0
            write_pressure = (write_time_delta / dt) > 450.0
        prev_disk = disk_now
        prev_time = now

        limits = _get_limits(limit_values)
        demand = limits["s1"] + limits["s2"] + limits["s3"]

        load_pressure = load_1m > (1.5 * cpu_total)
        cpu_pressure = cpu_pct > 80.0 or demand > budget
        mem_pressure = mem_pct > 85.0

        if not teardown_mode.is_set():
            if load_pressure:
                limits = {
                    "s1": max(1, limits["s1"] // 2),
                    "s2": max(1, limits["s2"] // 2),
                    "s3": max(1, limits["s3"] // 2),
                }

            if mem_pressure:
                stage1_pause_event.set()
                if limits["s2"] > 1:
                    limits["s2"] -= 1
            elif mem_pct < 80.0:
                stage1_pause_event.clear()

            if write_pressure and limits["s1"] > 1:
                limits["s1"] -= 1
            if read_pressure and limits["s2"] > 1:
                limits["s2"] -= 1

            limits = _clamp_limits_to_budget(limits, budget, cpu_total)

            if cpu_pressure:
                if limits["s2"] > 1:
                    limits["s2"] -= 1
                elif limits["s3"] > 1:
                    limits["s3"] -= 1
                limits = _clamp_limits_to_budget(limits, budget, cpu_total)
            else:
                healthy = (
                    (cpu_pct < 70.0)
                    and (mem_pct < 80.0)
                    and not read_pressure
                    and not write_pressure
                    and not load_pressure
                )
                if healthy:
                    # Keep stage3 prioritized for drain, then stage2, then stage1.
                    if limits["s3"] < base_limits["s3"]:
                        limits["s3"] += 1
                    elif limits["s2"] < base_limits["s2"]:
                        limits["s2"] += 1
                    elif limits["s1"] < base_limits["s1"]:
                        limits["s1"] += 1
                    limits = _clamp_limits_to_budget(limits, budget, cpu_total)

            _set_limits(limit_values, limits)

        stage3_pages = int(counters["stage3_pages"].value)
        pages_per_sec = (stage3_pages - prev_stage3_pages) / dt
        prev_stage3_pages = stage3_pages

        sample = {
            "timestamp": _ts(),
            "cpu_percent": round(cpu_pct, 3),
            "memory_percent": round(mem_pct, 3),
            "load_1m": round(load_1m, 3),
            "disk_read_time_delta_ms": round(read_time_delta, 3),
            "disk_write_time_delta_ms": round(write_time_delta, 3),
            "queue_sizes": {
                "pdf_queue": _queue_size_safe(queues["pdf"]),
                "image_queue": _queue_size_safe(queues["image"]),
                "block_queue": _queue_size_safe(queues["block"]),
                "metadata_queue": _queue_size_safe(queues["metadata"]),
            },
            "active_worker_limits": dict(limits),
            "totals": {
                "stage1_pages": int(counters["stage1_pages"].value),
                "stage2_pages": int(counters["stage2_pages"].value),
                "stage3_pages": int(counters["stage3_pages"].value),
                "stage2_blocks": int(counters["stage2_blocks"].value),
                "stage3_refined": int(counters["stage3_refined"].value),
            },
            "pages_processed": int(counters["stage3_pages"].value),
            "blocks_detected": int(counters["stage2_blocks"].value),
            "refined_blocks": int(counters["stage3_refined"].value),
            "pages_per_second": round(pages_per_sec, 4),
            "guardrails": {
                "cpu_pressure": bool(cpu_pressure),
                "memory_pressure": bool(mem_pressure),
                "load_pressure": bool(load_pressure),
                "disk_read_pressure": bool(read_pressure),
                "disk_write_pressure": bool(write_pressure),
                "stage1_paused": bool(stage1_pause_event.is_set()),
                "effective_core_budget": int(budget),
            },
        }
        _record_stats(stats_path, sample, max_samples=max_samples)

        if (now - last_progress_print) >= 5.0:
            print("[PIPELINE]")
            print(f"pages processed: {int(counters['stage3_pages'].value)}")
            print(f"blocks detected: {int(counters['stage2_blocks'].value)}")
            print(f"refined blocks: {int(counters['stage3_refined'].value)}")
            print(f"image_queue: {_queue_size_safe(queues['image'])}")
            print(f"block_queue: {_queue_size_safe(queues['block'])}")
            last_progress_print = now


def main() -> int:
    args = parse_args()
    if psutil is None:
        print("[stream][warn] psutil is not installed; adaptive CPU/memory/disk guardrails will run in degraded mode.")

    config = load_config(args.config)

    pdf_input_dir = get_path("pdf_input", config)
    images_output_dir = get_path("images_output", config)
    blocks_output_dir = get_path("blocks_output", config)
    refined_output_dir = get_path("refined_output", config)
    detections_output_dir = get_path("detections_output", config)

    print("[CONFIG]")
    print(f"pdf_input = {pdf_input_dir}")
    print(f"images_output = {images_output_dir}")
    print(f"blocks_output = {blocks_output_dir}")
    print(f"refined_output = {refined_output_dir}")
    print(f"detections_output = {detections_output_dir}")

    pdf_input_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir.mkdir(parents=True, exist_ok=True)
    blocks_output_dir.mkdir(parents=True, exist_ok=True)
    refined_output_dir.mkdir(parents=True, exist_ok=True)
    detections_output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in {pdf_input_dir}")
        return 0

    cpu_total = int(os.cpu_count() or 1)
    gpu_enabled = _gpu_available()
    if gpu_enabled:
        print("[PIPELINE] GPU detected, enabling GPU mode.")
        os.environ["PIPELINE_GPU_MODE"] = "1"

    base_limits = _compute_base_workers(cpu_total)
    base_limits = _apply_gpu_downscale(base_limits, gpu_enabled)
    cpu_budget = _effective_cpu_budget(cpu_total)
    init_limits = _clamp_limits_to_budget(base_limits, cpu_budget, cpu_total)
    print(
        f"[stream] cpu_total={cpu_total}, budget={cpu_budget}, "
        f"base_limits={base_limits}, init_limits={init_limits}"
    )

    ctx = mp.get_context("spawn")
    pdf_queue = ctx.Queue(maxsize=max(1, len(pdf_files)))
    image_queue = ctx.Queue(maxsize=max(10, int(args.image_queue_max)))
    block_queue = ctx.Queue(maxsize=max(10, int(args.block_queue_max)))
    metadata_queue = ctx.Queue(maxsize=max(50, int(args.metadata_queue_max)))

    for pdf_path in pdf_files:
        pdf_queue.put(str(pdf_path))

    stage1_input_closed = ctx.Event()
    stage2_input_closed = ctx.Event()
    stage3_input_closed = ctx.Event()
    metadata_input_closed = ctx.Event()
    stage1_pause_event = ctx.Event()
    stage1_input_closed.set()

    limit_values = {
        "s1": ctx.Value("i", int(init_limits["s1"])),
        "s2": ctx.Value("i", int(init_limits["s2"])),
        "s3": ctx.Value("i", int(init_limits["s3"])),
    }

    counters = {
        "stage1_pages": ctx.Value("i", 0),
        "stage2_pages": ctx.Value("i", 0),
        "stage3_pages": ctx.Value("i", 0),
        "stage2_blocks": ctx.Value("i", 0),
        "stage3_refined": ctx.Value("i", 0),
        "stage1_failures": ctx.Value("i", 0),
        "stage2_failures": ctx.Value("i", 0),
        "stage2_skipped": ctx.Value("i", 0),
        "metadata_writes": ctx.Value("i", 0),
        "metadata_failures": ctx.Value("i", 0),
    }

    stage1_workers = int(init_limits["s1"])
    stage2_workers = int(init_limits["s2"])
    stage3_workers = int(init_limits["s3"])

    processes = []

    metadata_proc = ctx.Process(
        target=_metadata_writer_worker,
        args=(metadata_queue, metadata_input_closed, str(detections_output_dir), counters),
        name="metadata-writer",
    )
    metadata_proc.start()
    processes.append(metadata_proc)

    s1_procs = []
    for idx in range(stage1_workers):
        p = ctx.Process(
            target=_stage1_worker,
            args=(
                idx,
                pdf_queue,
                stage1_input_closed,
                image_queue,
                str(images_output_dir),
                limit_values["s1"],
                stage1_pause_event,
                counters,
            ),
            name=f"stage1-{idx}",
        )
        p.start()
        s1_procs.append(p)
        processes.append(p)

    s2_procs = []
    for idx in range(stage2_workers):
        p = ctx.Process(
            target=_stage2_worker,
            args=(
                idx,
                image_queue,
                stage2_input_closed,
                block_queue,
                metadata_queue,
                str(blocks_output_dir),
                limit_values["s2"],
                counters,
                float(args.stage2_debug_sample_rate),
            ),
            name=f"stage2-{idx}",
        )
        p.start()
        s2_procs.append(p)
        processes.append(p)

    s3_procs = []
    for idx in range(stage3_workers):
        p = ctx.Process(
            target=_stage3_worker,
            args=(
                idx,
                block_queue,
                stage3_input_closed,
                metadata_queue,
                str(blocks_output_dir),
                str(refined_output_dir),
                str(detections_output_dir),
                limit_values["s3"],
                counters,
            ),
            name=f"stage3-{idx}",
        )
        p.start()
        s3_procs.append(p)
        processes.append(p)

    stats_path = Path(args.stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "started_at": _ts(),
                "mode": "streaming",
                "cpu_total": cpu_total,
                "base_limits": base_limits,
                "initial_limits": init_limits,
                "samples": [],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    controller_stop = threading.Event()
    teardown_mode = threading.Event()
    controller = threading.Thread(
        target=_controller_loop,
        args=(
            controller_stop,
            teardown_mode,
            limit_values,
            base_limits,
            cpu_budget,
            {"pdf": pdf_queue, "image": image_queue, "block": block_queue, "metadata": metadata_queue},
            counters,
            stage1_pause_event,
            stats_path,
            float(args.sample_interval),
            int(args.max_samples),
            cpu_total,
        ),
        daemon=True,
    )
    controller.start()

    start_time = time.time()
    controller_stopped = False
    try:
        for p in s1_procs:
            p.join()
        stage2_input_closed.set()
        for _ in range(stage2_workers):
            _safe_put(image_queue, None, timeout_seconds=60.0)

        # Keep limits stable during teardown to avoid workers getting throttled while exiting.
        teardown_mode.set()
        stage1_pause_event.clear()
        _set_limits(
            limit_values,
            {"s1": int(stage1_workers), "s2": int(stage2_workers), "s3": int(stage3_workers)},
        )

        for p in s2_procs:
            p.join()
        stage3_input_closed.set()
        for _ in range(stage3_workers):
            _safe_put(block_queue, None, timeout_seconds=60.0)

        for p in s3_procs:
            p.join()
        metadata_input_closed.set()
        _safe_put(metadata_queue, None, timeout_seconds=30.0)
        metadata_proc.join()
    finally:
        if not controller_stopped:
            controller_stop.set()
            controller.join(timeout=5.0)
            controller_stopped = True

    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)

    for q in (pdf_queue, image_queue, block_queue, metadata_queue):
        try:
            q.close()
        except Exception:
            pass
        try:
            q.join_thread()
        except Exception:
            pass

    failed_processes = {
        p.name: int(p.exitcode)
        for p in processes
        if p.exitcode not in (0, None)
    }

    elapsed = max(0.001, time.time() - start_time)
    stage3_pages = int(counters["stage3_pages"].value)
    final_summary = {
        "finished_at": _ts(),
        "elapsed_seconds": round(elapsed, 3),
        "totals": {
            "stage1_pages": int(counters["stage1_pages"].value),
            "stage2_pages": int(counters["stage2_pages"].value),
            "stage3_pages": stage3_pages,
            "stage2_blocks": int(counters["stage2_blocks"].value),
            "stage3_refined": int(counters["stage3_refined"].value),
            "stage1_failures": int(counters["stage1_failures"].value),
            "stage2_failures": int(counters["stage2_failures"].value),
            "stage2_skipped": int(counters["stage2_skipped"].value),
            "metadata_writes": int(counters["metadata_writes"].value),
            "metadata_failures": int(counters["metadata_failures"].value),
        },
        "pages_per_second": round(stage3_pages / elapsed, 4),
        "final_active_limits": _get_limits(limit_values),
        "failed_processes": failed_processes,
        "pages_processed": int(counters["stage3_pages"].value),
        "blocks_detected": int(counters["stage2_blocks"].value),
        "refined_blocks": int(counters["stage3_refined"].value),
    }

    _record_stats(
        stats_path,
        {
            "timestamp": _ts(),
            "final_summary": final_summary,
        },
        max_samples=max(1, int(args.max_samples)),
    )

    print("[stream] Stage1->Stage2->Stage3 completed")
    print(
        "[stream] "
        f"stage1_pages={final_summary['totals']['stage1_pages']} "
        f"stage2_pages={final_summary['totals']['stage2_pages']} "
        f"stage3_pages={final_summary['totals']['stage3_pages']} "
        f"pps={final_summary['pages_per_second']}"
    )
    if failed_processes:
        print(f"[stream][error] worker failures: {failed_processes}")
        return 1
    print(f"[stream] stats={stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
