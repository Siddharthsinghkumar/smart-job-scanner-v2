#!/bin/bash
echo "Timestamp, RAM_Used, GPU_Util, VRAM_Used, CPU_Util"
while true; do
    DATE=$(date +%H:%M:%S)
    RAM=$(free -m | awk '/Mem:/ { print $3 }')
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr ',' ' ')
    CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    echo "$DATE, ${RAM}MB, $GPU, ${CPU}%"
    sleep 1
done
