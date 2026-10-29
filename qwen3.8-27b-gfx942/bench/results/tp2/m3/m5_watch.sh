#!/bin/bash
# Disposable: runs on the Alola login node (tmux m5-watch). When a held M5 job (name
# mktp2b: second node, mktp2d: second day) starts, run plans/m5.sh in it from the frozen
# source $R/src-m5, then cancel the job.
R=/cluster/logs/qwen38-tp2
done_jobs=" "
while true; do
  for line in $(squeue -h -u $USER -t R -o "%i:%j:%N" | grep -E ":mktp2[bd]:"); do
    j=${line%%:*}; rest=${line#*:}; n=${rest#*:}; name=${rest%%:*}
    case "$done_jobs" in *" $j "*) continue ;; esac
    done_jobs="$done_jobs$j "
    out=$R/m5_${name}_${n}_$(date +%Y%m%d)
    mkdir -p $out && cp $R/src-m5/scratch/m3/plans/m5.sh $out/plan.sh
    echo "$(date) job $j ($name) on $n -> $out" >> $R/m5_watch.log
    ( JOB=$j OUT=$out SRC=$R/src-m5 bash $R/src-m5/scratch/m3/plan_lib.sh $out/plan.sh > $out/driver.log 2>&1
      echo "$(date) job $j done" >> $R/m5_watch.log; scancel $j ) &
  done
  sleep 60
done
