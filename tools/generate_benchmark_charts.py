
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_charts(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    res1 = data.get("benchmark_1")
    res13 = data.get("benchmark_13")
    
    if not res1:
        print("Missing benchmark_1 data")
        return

    output_dir = Path("artifacts/benchmark_charts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage-time breakdown chart
    if res13:
        labels = ['Stage 1', 'Stage 2', 'Stage 3']
        s1 = [res1['s1_time'], res13['s1_time']]
        s2 = [res1['s2_time'], res13['s2_time']]
        s3 = [res1['s3_time'], res13['s3_time']]
        
        x = np.arange(len(['1 PDF', '13 PDFs']))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, s1, width, label='Stage 1 (PDF->Img)')
        ax.bar(x, s2, width, label='Stage 2 (Detection)')
        ax.bar(x + width, s3, width, label='Stage 3 (OCR)')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Stage-wise Runtime Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['1 PDF', '13 PDFs'])
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "stage_breakdown.png")
        plt.close()
    else:
        # Single run breakdown
        labels = ['Stage 1', 'Stage 2', 'Stage 3']
        times = [res1['s1_time'], res1['s2_time'], res1['s3_time']]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(labels, times)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Stage-wise Runtime (1 PDF)')
        plt.tight_layout()
        plt.savefig(output_dir / "stage_breakdown_1.png")
        plt.close()

    # 2. Throughput comparison chart
    if res13:
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.25
        x = np.arange(len(['1 PDF', '13 PDFs']))
        p_sec = [res1['pages_per_sec'], res13['pages_per_sec']]
        c_sec = [res1['crops_per_sec'], res13['crops_per_sec']]
        
        ax.bar(x - width/2, p_sec, width, label='Pages/sec')
        ax.bar(x + width/2, c_sec, width, label='Crops/sec')
        
        ax.set_ylabel('Throughput')
        ax.set_title('Throughput Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['1 PDF', '13 PDFs'])
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "throughput.png")
        plt.close()

    # 3. RAM/VRAM usage chart over time
    for name, res in [("1_PDF", res1), ("13_PDF", res13)]:
        if not res: continue
        history = res.get("history", [])
        if not history: continue
        
        times = [h['time'] - history[0]['time'] for h in history]
        ram = [h['ram'] for h in history]
        vram = [h['vram'] for h in history]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('RAM (MB)', color='tab:blue')
        ax1.plot(times, ram, color='tab:blue', label='RAM Usage')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('VRAM (MB)', color='tab:red')
        ax2.plot(times, vram, color='tab:red', label='VRAM Usage')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title(f'Resource Usage Over Time ({name})')
        fig.tight_layout()
        plt.savefig(output_dir / f"resource_usage_{name}.png")
        plt.close()

    # 4. OCR mode chart
    for name, res in [("1_PDF", res1), ("13_PDF", res13)]:
        if not res: continue
        total = res['total_crops']
        downscaled = res['downscaled_count']
        tiled = res['tiled_count']
        normal = max(0, total - downscaled - tiled)
        
        labels = ['Normal', 'Downscaled', 'Tiled']
        values = [normal, downscaled, tiled]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title(f'OCR Modes ({name})')
        plt.savefig(output_dir / f"ocr_modes_{name}.png")
        plt.close()

    print(f"Charts generated in {output_dir}")

if __name__ == "__main__":
    generate_charts("benchmark_results.json")
