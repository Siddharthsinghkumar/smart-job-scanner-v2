# Hardware Resource Mandate

## CPU Usage Policy
- **Maximum Utilization**: No more than 80% of total logical cores may be utilized for CPU-intensive tasks combined.
- **20-Core System Limit**: 16 logical cores total for CPU-bound processes.
- **CPU Process Allocation (Stage 2)**:
  - CPU Producers (Rendering/Slicing): 10
  - CPU Stitchers (Merging): 4
  - Feeder/Managers: 2
  - **Total CPU Cores**: 16

## GPU Usage Policy
- **GPU Workers**: Independent of CPU core limits.
- **Constraint**: Governed by available VRAM (4GB) and GPU compute saturation.
- **Allocation**: 4 GPU Workers.

## Enforcement
- All future optimizations must stay within the 16-core CPU boundary.
- GPU workers must be monitored for VRAM OOM risk.
