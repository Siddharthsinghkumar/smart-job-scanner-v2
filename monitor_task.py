import sys, time, os, psutil, subprocess

def monitor(runner_pid, log_file):
    print(f"--- Standalone Sentry Active (Watching PID: {runner_pid}) ---")
    last_log_size = 0
    inactivity_count = 0
    start_time = time.time()
    
    while True:
        try:
            # 1. Check if runner is still alive
            process = psutil.Process(runner_pid)
        except psutil.NoSuchProcess:
            print("--- Sentry: Runner finished normally. ---")
            break

        # 2. Check Hardware Activity
        try:
            cpu_usage = process.cpu_percent(interval=1)
            gpu_res = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
            gpu_util, vram = map(int, gpu_res.strip().split(","))
        except:
            cpu_usage, gpu_util, vram = 0, 0, 0

        # 3. Check Log Growth
        log_exists = os.path.exists(log_file)
        if log_exists:
            current_size = os.path.getsize(log_file)
            if current_size == last_log_size:
                # Log hasn't moved. Check if we are really stuck.
                if cpu_usage < 15.0 and gpu_util == 0:
                    inactivity_count += 1
            else:
                inactivity_count = 0
                last_log_size = current_size
        else:
            # Log missing for more than 15s after startup
            if time.time() - start_time > 15:
                inactivity_count += 1

        print(f"[Sentry] CPU: {cpu_usage:.1f}% | GPU: {gpu_util}% | VRAM: {vram}MB | Inactive: {inactivity_count * 5}s")

        # 4. KILL MANDATE: 45 seconds of zero activity
        if inactivity_count >= 9: 
            print(f"!!! SENTRY KILL: PID {runner_pid} IS DOING NOTHING !!!")
            # Kill process and all children
            parent = psutil.Process(runner_pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            break

        time.sleep(5)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: monitor_task.py <pid> <log_file>")
    else:
        monitor(int(sys.argv[1]), sys.argv[2])
