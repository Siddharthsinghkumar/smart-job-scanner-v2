#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import sys
import numpy as np

def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    
    run_info = {}
    if (run_dir / "run_info.json").exists():
        with open(run_dir / "run_info.json", "r") as f:
            run_info = json.load(f)

    events = []
    if (run_dir / "events.jsonl").exists():
        with open(run_dir / "events.jsonl", "r") as f:
            for line in f: events.append(json.loads(line))
        
    stages = ["stage1", "stage2", "stage3", "stage4"]
    timings = {}
    stage_metadata = {}
    for s in stages:
        starts = [e["ts"] for e in events if e["event"] == "stage_start" and e["stage"] == s]
        ends = [e["ts"] for e in events if e["event"] == "stage_end" and e["stage"] == s]
        if starts and ends:
            timings[s] = ends[0] - starts[0]
        
        meta = [e["metadata"] for e in events if e["event"] == "metadata" and e.get("metadata", {}).get("stage") == s]
        if meta:
            stage_metadata[s] = meta[0]
            
    metrics = {}
    for s in ["stage2", "stage3", "stage4"]:
        m_file = run_dir / f"metrics_{s}.json"
        if m_file.exists():
            with open(m_file, "r") as f:
                metrics[s] = json.load(f)

    resources = []
    if (run_dir / "resources.jsonl").exists():
        with open(run_dir / "resources.jsonl", "r") as f:
            for line in f:
                try: resources.append(json.loads(line))
                except: pass

    res_stats = {}
    working_idle = {}
    for s in stages:
        starts = [e["ts"] for e in events if e["event"] == "stage_start" and e["stage"] == s]
        ends = [e["ts"] for e in events if e["event"] == "stage_end" and e["stage"] == s]
        if not starts or not ends: continue
        t0, t1 = starts[0], ends[0]
        
        window = [r for r in resources if t0 <= r["ts"] <= t1]
        if not window: continue
        
        cpu = [r["cpu_percent"] for r in window if r["cpu_percent"] is not None]
        ram = [r["ram_used_mb"] for r in window if r["ram_used_mb"] is not None]
        gpu = [r["gpu_util"] for r in window if r["gpu_util"] is not None]
        vram = [r["vram_used_mb"] for r in window if r["vram_used_mb"] is not None]
        
        # IOPS calculation
        read_counts = [r["disk_read_count"] for r in window if r["disk_read_count"] is not None]
        write_counts = [r["disk_write_count"] for r in window if r["disk_write_count"] is not None]
        
        riops = []
        wiops = []
        if len(window) > 1:
            for i in range(1, len(window)):
                dt = window[i]["ts"] - window[i-1]["ts"]
                if dt > 0:
                    riops.append((window[i]["disk_read_count"] - window[i-1]["disk_read_count"]) / dt)
                    wiops.append((window[i]["disk_write_count"] - window[i-1]["disk_write_count"]) / dt)

        # Working vs Idle
        working_samples = 0
        for r in window:
            is_working = (r["cpu_percent"] or 0) > 5 or (r["gpu_util"] or 0) > 5
            if is_working: working_samples += 1
        
        total_samples = len(window)
        duration = t1 - t0
        working_time = (working_samples / total_samples) * duration if total_samples > 0 else 0
        working_idle[s] = {"working": working_time, "idle": duration - working_time}

        res_stats[s] = {
            "cpu_avg": np.mean(cpu) if cpu else 0,
            "cpu_peak": np.max(cpu) if cpu else 0,
            "cpu_p1": np.percentile(cpu, 1) if cpu else 0,
            "ram_avg": np.mean(ram) if ram else 0,
            "ram_peak": np.max(ram) if ram else 0,
            "gpu_avg": np.mean(gpu) if gpu else 0,
            "gpu_peak": np.max(gpu) if gpu else 0,
            "vram_avg": np.mean(vram) if vram else 0,
            "vram_peak": np.max(vram) if vram else 0,
            "disk_read_iops_avg": np.mean(riops) if riops else 0,
            "disk_write_iops_avg": np.mean(wiops) if wiops else 0,
            "disk_read_iops_peak": np.max(riops) if riops else 0,
            "disk_write_iops_peak": np.max(wiops) if wiops else 0
        }

    total_time = sum(timings.values())
    
    # Validation Rules
    s2_m = metrics.get("stage2", {})
    s3_m = metrics.get("stage3", {})
    s4_m = metrics.get("stage4", {})
    
    rejections = []
    if s2_m.get("tp", 0) < 46: rejections.append("TP < 46")
    if s2_m.get("fp", 0) > 5000: rejections.append("FP > 5000")
    raw_dets = stage_metadata.get("stage2", {}).get("raw_detections", 0)
    if raw_dets > 10000: rejections.append("raw_detections > 10000")
    if timings.get("stage2", 0) > 60: rejections.append("Stage 2 time too high (>60s)")
    boxes_ocr = stage_metadata.get("stage3", {}).get("boxes_into_ocr", 0)
    if boxes_ocr > 500: rejections.append("OCR load too large (>500 boxes)")
    
    verdict = "PASS" if not rejections else f"REJECT ({', '.join(rejections)})"

    md = []
    md.append(f"# Benchmark Report: {run_info.get('run_id', 'N/A')}\n")
    
    md.append("### 1) Run summary\n")
    md.append(f"* **Run ID:** {run_info.get('run_id', 'N/A')}")
    md.append(f"* **PDF:** {run_info.get('pdf', 'N/A')}")
    md.append(f"* **Total Time:** {total_time:.2f} s")
    md.append(f"* **Final TP:** {s4_m.get('tp', 0)}")
    md.append(f"* **Final FP:** {s4_m.get('fp', 0)}")
    md.append(f"* **Final FN:** {s4_m.get('fn', 0)}")
    md.append(f"* **Final GT:** {s4_m.get('gt_total', 0)}")
    md.append(f"* **Final Recall:** {s4_m.get('recall', 0):.3f}")
    md.append(f"* **Final Precision:** {s4_m.get('precision', 0):.3f}\n")

    md.append("### 2) Main stage table\n")
    md.append("| Stage | Time (s) | TP | FP | FN | Recall | Precision |")
    md.append("|-------|----------|----|----|----|--------|-----------|")
    for s in stages:
        t = timings.get(s, 0)
        m = metrics.get(s, {})
        md.append(f"| {s} | {t:.2f} | {m.get('tp','-')} | {m.get('fp','-')} | {m.get('fn','-')} | {format(m.get('recall'), '.3f') if 'recall' in m else '-'} | {format(m.get('precision'), '.3f') if 'precision' in m else '-'} |")
    md.append("")

    md.append("### 3) Stage 2 deep report\n")
    s2_meta = stage_metadata.get("stage2", {})
    md.append(f"* **Stage:** stage2")
    md.append(f"* **Wall Time:** {timings.get('stage2', 0):.2f} s")
    md.append(f"* **TP:** {s2_m.get('tp', 0)}")
    md.append(f"* **FP:** {s2_m.get('fp', 0)}")
    md.append(f"* **FN:** {s2_m.get('fn', 0)}")
    md.append(f"* **GT:** {s2_m.get('gt_total', 0)}")
    md.append(f"* **Recall:** {s2_m.get('recall', 0):.3f}")
    md.append(f"* **Precision:** {s2_m.get('precision', 0):.3f}")
    md.append(f"* **Raw Detections:** {s2_meta.get('raw_detections', 0)}")
    md.append(f"* **Post-NMS Detections:** {s2_meta.get('post_nms_detections', 0)}")
    md.append(f"* **Tile Count:** {s2_meta.get('tile_count', 0)}")
    avg_boxes = s2_meta.get('raw_detections', 0) / s2_meta.get('tile_count', 1) if s2_meta.get('tile_count', 0) > 0 else 0
    md.append(f"* **Avg Boxes per Tile:** {avg_boxes:.2f}\n")

    md.append("### 4) Resource report per stage\n")
    md.append("| Stage | CPU avg | CPU peak | CPU P1 | RAM avg (MB) | RAM peak (MB) | GPU avg | GPU peak | VRAM avg (MB) | VRAM peak (MB) |")
    md.append("|-------|---------|----------|--------|--------------|---------------|---------|----------|---------------|----------------|")
    for s in stages:
        rs = res_stats.get(s, {})
        if rs:
            md.append(f"| {s} | {rs['cpu_avg']:.1f}% | {rs['cpu_peak']:.1f}% | {rs['cpu_p1']:.1f}% | {rs['ram_avg']:.0f} | {rs['ram_peak']:.0f} | {rs['gpu_avg']:.1f}% | {rs['gpu_peak']:.1f}% | {rs['vram_avg']:.0f} | {rs['vram_peak']:.0f} |")
    
    md.append("\n**Disk Telemetry:**\n")
    md.append("| Stage | Read IOPS Avg | Write IOPS Avg | Read IOPS Peak | Write IOPS Peak |")
    md.append("|-------|---------------|----------------|----------------|-----------------|")
    for s in stages:
        rs = res_stats.get(s, {})
        if rs:
            md.append(f"| {s} | {rs['disk_read_iops_avg']:.1f} | {rs['disk_write_iops_avg']:.1f} | {rs['disk_read_iops_peak']:.1f} | {rs['disk_write_iops_peak']:.1f} |")
    md.append("")

    md.append("### 5) Stage-to-stage load impact\n")
    md.append(f"* **Stage2 to Stage3:** Transition")
    md.append(f"* **Boxes into OCR:** {boxes_ocr}")
    page_count = run_info.get("page_count", 0)
    avg_per_page = boxes_ocr / page_count if page_count > 0 else boxes_ocr
    md.append(f"* **Avg Boxes per Page:** {avg_per_page:.2f}")
    md.append(f"* **Expected OCR Time:** {boxes_ocr * 0.5:.2f} s (estimated at 0.5s/box)\n")

    md.append("### 6) Working vs idle time\n")
    md.append("| Stage | Working Time (s) | Idle Time (s) |")
    md.append("|-------|------------------|---------------|")
    for s in stages:
        wi = working_idle.get(s, {"working": 0, "idle": 0})
        md.append(f"| {s} | {wi['working']:.2f} | {wi['idle']:.2f} |")
    md.append("")

    md.append("### 7) Acceptance / rejection rules\n")
    md.append(f"* **Verdict:** {verdict}")
    if rejections:
        for r in rejections:
            md.append(f"  - ❌ {r}")
    else:
        md.append("  - ✅ All criteria met")
    md.append("")

    md.append("### 8) Comparison summary (Single Run)\n")
    md.append("| Config | TP | FP | Raw Boxes | Time | Verdict |")
    md.append("|--------|----|----|-----------|------|---------|")
    md.append(f"| Current | {s4_m.get('tp',0)} | {s4_m.get('fp',0)} | {raw_dets} | {total_time:.2f} | {verdict} |\n")

    with open(run_dir / "final_report.md", "w") as f:
        f.write("\n".join(md))

    report = {
        "run_info": run_info,
        "timing": timings,
        "metrics": metrics,
        "resources": res_stats,
        "metadata": stage_metadata,
        "working_idle": working_idle,
        "verdict": verdict,
        "rejections": rejections
    }
    report = to_python_types(report)
    with open(run_dir / "final_report.json", "w") as f:
        json.dump(report, f)

if __name__ == "__main__":
    main()
