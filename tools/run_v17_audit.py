import subprocess, time, os, psutil, threading

def sentry_v17_2(pid, log_file, stop_event):
    print("--- Iron Sentry v17.2 Active ---")
    zombie_timer = 0
    
    while not stop_event.is_set():
        # 1. Query Hardware
        try:
            gpu_res = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
            gpu_util, vram = map(int, gpu_res.strip().split(","))
        except:
            gpu_util, vram = 0, 0

        # 2. THE ZOMBIE GUARD
        # If GPU compute is 0 but VRAM is high (>500MB), start counting
        if gpu_util == 0 and vram > 500:
            zombie_timer += 5
        else:
            zombie_timer = 0
            
        if zombie_timer >= 30:
            print(f"!!! SENTRY KILL: GPU DEADLOCK (Util: {gpu_util}%, VRAM: {vram}MB) !!!")
            os.killpg(os.getpgid(pid), 9)
            break

        # 3. RAM Safety
        mem = psutil.virtual_memory()
        if mem.percent > 95:
            print(f"!!! SENTRY KILL: RAM {mem.percent}% !!!")
            os.killpg(os.getpgid(pid), 9)
            break
            
        time.sleep(5)

def run_production():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ""
    log_file = "victory_v17.log"
    if os.path.exists(log_file): os.remove(log_file)
    
    cmd = ["./4_env/bin/python", "scripts/run_supersonic_v16_6.py", "--dir", "data/final_audit"]
    start = time.time()
    
    with open(log_file, "w") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
    
    stop_ev = threading.Event()
    t = threading.Thread(target=sentry_v17_2, args=(p.pid, log_file, stop_ev))
    t.start()
    
    try:
        p.wait()
        print(f"\n🏁 S1-S4 Complete. Wall Time: {time.time()-start:.2f}s")
    finally:
        stop_ev.set()
        t.join()

if __name__ == "__main__":
    run_production()
