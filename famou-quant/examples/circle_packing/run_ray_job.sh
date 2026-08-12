#!/bin/bash
# =============================================================================
# Circle Packing — Ray Job 提交
#
# 使用方式:
#   bash examples/circle_packing/run_ray_job.sh
#
# 可通过环境变量覆盖默认值:
#   RAY_ADDRESS=http://10.94.147.104:8265 bash examples/circle_packing/run_ray_job.sh
#   JOB_NAME=my-test bash examples/circle_packing/run_ray_job.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export RAY_ADDRESS="${RAY_ADDRESS:-http://10.94.147.104:8265}"
export JOB_NAME="${JOB_NAME:-circlepacking}"

python "${PROJECT_ROOT}/run_famou_ray_job.py" \
  --config  "${SCRIPT_DIR}/config_ray.yaml" \
  --program "${SCRIPT_DIR}/init.py" \
  --evaluator "${SCRIPT_DIR}/evaluator.py"
