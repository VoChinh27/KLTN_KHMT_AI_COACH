#!/usr/bin/env bash
# Chạy toàn bộ video trong Input/ -> Output/. Chạy từ thư mục gốc dự án:  bash scripts/run_demo.sh
set -e
PY=${PYTHON:-python3}
$PY -m badminton3d run
echo "Xong. Mở Output/_tong_hop/index.html (tổng hợp) hoặc Output/<video>/report.html (từng video)."
