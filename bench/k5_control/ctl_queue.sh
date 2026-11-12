#!/bin/bash
# ctl_queue.sh BIN -- K5_CONTROL's arms, one at a time (short, then long), each started only with >= 16 GB available.
BIN=$1
run() {
  local label=$1; shift
  while :; do
    av=$(free -g | awk '/^Mem:/{print $7}')
    [ "$av" -ge 16 ] && break
    sleep 30
  done
  ~/ctl_arm.sh "$BIN" "$label" "$@"
  echo "$(cat ~/ctl_runs/$label.done)" >> ~/ctl_runs/queue.log
}
echo "queue start $(date +%T)" >> ~/ctl_runs/queue.log
run CL --k5-control link=1
run CL_long --long --k5-control link=1
run CW --k5-control words=1
run CW_long --long --k5-control words=1
run CLW --k5-control link=1,words=1
run CLW_long --long --k5-control link=1,words=1
echo "queue end $(date +%T)" >> ~/ctl_runs/queue.log
touch ~/ctl_runs/queue.done
