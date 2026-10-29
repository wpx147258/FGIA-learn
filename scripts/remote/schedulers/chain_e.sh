#!/usr/bin/env bash
for _ in $(seq 1 600); do
  grep -q "ALL_PHASES_DONE" /root/restore_plan.log 2>/dev/null && break
  sleep 60
done
bash /root/restore_phaseE.sh >> /root/restore_phaseE.log 2>&1
