#!/bin/bash
# Start or resume the pipeline driver (driver.py) in tmux session mkpipe on the Alola login
# node. Idempotent: `start` does nothing while the session lives. A user crontab line runs
# `start` every 10 minutes, so the driver comes back after a lost tmux server or a login-node
# reboot; the state in ~/mk/out/pipeline/state makes it resume. After ALL_DONE, `start`
# removes the crontab line.
# Usage: bash ~/mk/pipeline/run.sh start|stop|loop
P=${PIPE_DIR:-$HOME/mk/pipeline}
O=${PIPE_OUT:-$HOME/mk/out/pipeline}
S=${PIPE_TMUX:-mkpipe}
CRON="*/10 * * * * bash $P/run.sh start > /dev/null 2>&1"
case ${1:-start} in
start)
  if [ -e "$O/state/ALL_DONE" ]; then
    crontab -l 2>/dev/null | grep -vF "$P/run.sh start" | crontab -
    echo "ALL_DONE: nothing to start"; exit 0
  fi
  crontab -l 2>/dev/null | grep -qF "$P/run.sh start" || { (crontab -l 2>/dev/null; echo "$CRON") | crontab -; }
  if tmux has-session -t "$S" 2>/dev/null; then echo "tmux $S runs"; exit 0; fi
  mkdir -p "$O/state"
  tmux new-session -d -s "$S" "bash $P/run.sh loop"
  echo "started tmux $S on $(hostname); log $O/driver.log"
  ;;
stop)  # stop the driver and the watchdog; running stages continue, state is kept
  crontab -l 2>/dev/null | grep -vF "$P/run.sh start" | crontab -
  tmux kill-session -t "$S" 2>/dev/null
  echo "stopped"
  ;;
loop)  # restart the driver after an exit until ALL_DONE
  while [ ! -e "$O/state/ALL_DONE" ]; do
    python3 "$P/driver.py" >> "$O/driver.log" 2>&1
    echo "driver exit $? $(date -Is)" >> "$O/driver.log"
    [ -e "$O/state/ALL_DONE" ] || sleep 60
  done
  ;;
esac
