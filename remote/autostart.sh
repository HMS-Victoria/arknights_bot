#!/bin/bash
set -u

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

cd /root/arknights || exit 1
LOG=/root/autostart.log

{
  echo "[autostart] start $(date)"
  if ! /root/miniconda3/bin/pip list 2>/dev/null | grep -q '^trl '; then
    echo "[autostart] installing requirements"
    /root/miniconda3/bin/pip install -i "$PIP_INDEX_URL" -r /root/arknights/train/requirements.txt
  else
    echo "[autostart] requirements already installed"
  fi

  echo "[autostart] starting training"
  nohup bash /root/arknights/train/run_train.sh > /root/train.log 2>&1 &
  echo $! > /root/train.pid
  echo "[autostart] training pid $(cat /root/train.pid)"
} >> "$LOG" 2>&1
