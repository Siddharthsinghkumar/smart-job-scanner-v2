#!/bin/bash
START_TIME=$(date +%s)
echo ">>> UNIFIED SESSION STARTED (PID: $$) <<<"
./4_env/bin/python scripts/run_supersonic_v16_6.py --dir data/raw_pdfs > session_log.txt 2>&1 &
RUNNER_PID=$!
./4_env/bin/python monitor_task.py $RUNNER_PID session_log.txt &
MONITOR_PID=$!

wait $RUNNER_PID
EXIT_CODE=$?

kill $MONITOR_PID 2>/dev/null

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Unified Session Finished. Total Wall Time: ${ELAPSED}s"
else
    echo "❌ Unified Session Failed (Exit Code: $EXIT_CODE). Total Wall Time: ${ELAPSED}s"
fi
