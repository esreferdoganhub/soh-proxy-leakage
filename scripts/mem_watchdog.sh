#!/bin/zsh
# Benchmark Python işleminin fiziksel belleğini izler; LIMIT_MB'ı aşarsa durdurur (ledger kaldığı yerden devam eder).
LIMIT_MB=${1:-6000}; LOG=${2:-results/mem_watchdog.log}
while true; do
  P=""; for c in $(pgrep -f "scripts/mit_tuned_benchmark.py"); do [ "$(ps -o comm= -p $c | xargs basename)" = "Python" ] && P=$c; done
  [ -z "$P" ] && { echo "$(date '+%F %T') benchmark yok, bekçi çıkıyor" >> $LOG; exit 0; }
  MB=$(footprint -p $P 2>/dev/null | awk '/Footprint:/{for(i=1;i<=NF;i++) if($i=="Footprint:"){v=$(i+1);u=$(i+2)}; if(u=="KB")v=v/1024; if(u=="GB")v=v*1024; printf "%d", v; exit}')
  SW=$(sysctl -n vm.swapusage | awk '{print $6}')
  echo "$(date '+%F %T') pid=$P footprint=${MB}MB swap_used=$SW" >> $LOG
  if [ -n "$MB" ] && [ "$MB" -gt "$LIMIT_MB" ]; then echo "$(date '+%F %T') LİMİT AŞILDI → durduruluyor" >> $LOG; pkill -f "scripts/mit_tuned_benchmark.py"; exit 1; fi
  sleep 120
done
