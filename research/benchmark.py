import subprocess
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import socket
import psutil
import threading
from statistics import mean

# --- Configuration ---
CPU_THREADS_TO_TEST = [None] + list(range(4, 21))  # GPU (None) + CPU threads 4→20
MODEL = "mistral:latest"
BENCHMARK_PROMPT = "Write a 500-word essay about the history of artificial intelligence, from its early theoretical stages to modern-day large language models."
WARMUP_PROMPTS = ["hello", "write a short poem", "tell me a fun fact"]

# --- Utility Functions ---
def wait_for_server(port=11434, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    raise TimeoutError("Ollama server did not start in time.")

def sample_metrics(stop_event, cpu_list, gpu_list):
    """Thread: sample CPU% and GPU power every second."""
    while not stop_event.is_set():
        # CPU usage (%)
        cpu_list.append(psutil.cpu_percent(interval=None))
        # GPU power (watts)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            power = float(result.stdout.strip().split("\n")[0])
            gpu_list.append(power)
        except Exception:
            gpu_list.append(0.0)
        time.sleep(1)

# --- Main Logic ---
results = []
log_dir = "benchmark_logs"
os.makedirs(log_dir, exist_ok=True)

for threads in CPU_THREADS_TO_TEST:
    test_name = f"CPU ({threads} Threads)" if threads is not None else "GPU"
    print(f"\n--- Starting benchmark for: {test_name} ---")

    # Kill existing Ollama processes
    subprocess.run(["pkill", "ollama"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

    # Start server with environment variables
    env_vars = os.environ.copy()
    if threads is not None:
        env_vars["OLLAMA_NO_GPU"] = "1"
        env_vars["OLLAMA_NUM_THREADS"] = str(threads)
    else:
        env_vars.pop("OLLAMA_NO_GPU", None)
        env_vars.pop("OLLAMA_NUM_THREADS", None)

    server_process = subprocess.Popen(
        ["ollama", "serve"],
        env=env_vars,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait until server is ready
    try:
        wait_for_server()
    except TimeoutError:
        print("❌ Server did not start, skipping test.")
        server_process.terminate()
        continue

    # Warm-up prompts
    print("Warming up the model...")
    for prompt in WARMUP_PROMPTS:
        subprocess.run(["ollama", "run", MODEL, prompt], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Run benchmark with realtime streaming
    print("Running main benchmark...\n")
    start_time = time.time()

    cpu_samples, gpu_samples = [], []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=sample_metrics, args=(stop_event, cpu_samples, gpu_samples))
    monitor_thread.start()

    process = subprocess.Popen(
        ["ollama", "run", MODEL, "--verbose", BENCHMARK_PROMPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_vars
    )

    log_file_path = os.path.join(log_dir, f"{test_name.replace(' ', '_').replace('(', '').replace(')', '')}.log")
    with open(log_file_path, "w") as log_file:
        for line in process.stdout:
            print(line, end="")        # realtime feedback
            log_file.write(line)       # save log

    process.wait()
    stop_event.set()
    monitor_thread.join()
    wall_time = time.time() - start_time

    # Extract metrics
    with open(log_file_path, "r") as f:
        log_text = f.read()

    eval_rate_match = re.search(r"eval rate:\s+([\d.]+)\s+tokens/s", log_text)
    duration_match = re.search(r"total duration:\s+([\d.]+)", log_text)
    tokens_match = re.search(r"output tokens:\s+(\d+)", log_text)

    tokens = int(tokens_match.group(1)) if tokens_match else 0
    total_duration = float(duration_match.group(1)) if duration_match else wall_time

    # If Ollama reported eval rate, use it, otherwise compute
    if eval_rate_match:
        eval_rate = float(eval_rate_match.group(1))
    elif tokens > 0 and wall_time > 0:
        eval_rate = tokens / wall_time
    else:
        eval_rate = 0.0

    avg_cpu = mean(cpu_samples) if cpu_samples else 0.0
    avg_gpu_power = mean(gpu_samples) if gpu_samples else 0.0

    results.append({
        "Config": test_name,
        "Eval Rate (tokens/s)": eval_rate,
        "Total Duration (s)": total_duration,
        "Output Tokens": tokens,
        "Avg CPU (%)": avg_cpu,
        "Avg GPU Power (W)": avg_gpu_power,
        "Wall Time (s)": wall_time
    })

    print(f"\n✅ Benchmark complete: {test_name}")
    print(f"Eval Rate: {eval_rate:.2f} tokens/s | Duration: {total_duration:.2f}s | Tokens: {tokens} | "
          f"Avg CPU: {avg_cpu:.1f}% | Avg GPU Power: {avg_gpu_power:.1f} W | Wall Time: {wall_time:.2f}s")

    # Stop server
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()

    time.sleep(5)

print("\n--- All benchmarks complete! ---")
df = pd.DataFrame(results)
df_sorted = df.sort_values(by="Eval Rate (tokens/s)", ascending=False)

# Save summary CSV
summary_csv = os.path.join(log_dir, "summary.csv")
df_sorted.to_csv(summary_csv, index=False)
print(f"\nSummary saved to {summary_csv}\n")
print(df_sorted.to_string(index=False))

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')

# Chart 1: Eval Rate
fig1, ax1 = plt.subplots(figsize=(14, 8))
colors = ['#1f77b4' if 'CPU' in config else '#ff7f0e' for config in df_sorted['Config']]
bars = ax1.bar(df_sorted['Config'], df_sorted['Eval Rate (tokens/s)'], color=colors)

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
             f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_title(f'Ollama Performance Comparison on {MODEL} (Eval Rate)', fontsize=18, pad=20)
ax1.set_xlabel('Configuration', fontsize=14)
ax1.set_ylabel('Eval Rate (tokens/s)', fontsize=14)
ax1.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('ollama_benchmark_eval_rate.png')
print("Chart 1 saved as 'ollama_benchmark_eval_rate.png'")

# Chart 2: Wall Time
fig2, ax2 = plt.subplots(figsize=(14, 8))
bars2 = ax2.bar(df_sorted['Config'], df_sorted['Wall Time (s)'], color=colors)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
             f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_title(f'Ollama Performance Comparison on {MODEL} (Wall Time)', fontsize=18, pad=20)
ax2.set_xlabel('Configuration', fontsize=14)
ax2.set_ylabel('Wall Time (s)', fontsize=14)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('ollama_benchmark_wall_time.png')
print("Chart 2 saved as 'ollama_benchmark_wall_time.png'")

plt.show()
