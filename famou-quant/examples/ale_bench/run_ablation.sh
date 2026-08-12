#!/bin/bash
# ============================================================================
#  经验检索消融实验 (Ablation: Experience Retrieval)
#
#  实验设计：
#    - 两组实验，唯一变量是 experience_retrieval.enabled
#    - 两组都使用 ExperienceJudge，都开启 debug loop
#    - ablation_no_retrieval : ExperienceJudge 产出经验但不注入（baseline）
#    - ablation_with_retrieval: ExperienceJudge 产出经验并注入（treatment）
#
#  共 5 tasks × 2 groups = 10 个实验
#  模型: deepseek-v3.2, 迭代数: 100, debug_max_iterations: 3
# ============================================================================

set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EXAMPLES_DIR="${ROOT_DIR}/examples/ale_bench"
TASKS="ahc008 ahc015 ahc016 ahc025 ahc046"

echo "============================================"
echo "  经验检索消融实验"
echo "  ROOT: ${ROOT_DIR}"
echo "============================================"
echo ""

# --- Group 1: baseline (no retrieval) ---
echo ">>> [Group 1/2] ablation_no_retrieval (ExperienceJudge ON, retrieval OFF)"
echo ""
for task in ${TASKS}; do
    TASK_DIR="${EXAMPLES_DIR}/${task}"
    CONFIG="${TASK_DIR}/config_ablation_no_retrieval.yaml"
    INIT="${TASK_DIR}/init.cpp"
    EVALUATOR="${TASK_DIR}/evaluator.py"

    echo "  Starting: ${task} (no_retrieval)..."
    python "${ROOT_DIR}/run_famou.py" \
        -c "${CONFIG}" \
        -p "${INIT}" \
        -e "${EVALUATOR}" &
    echo "    PID: $!"
done

echo ""
echo ">>> Waiting for Group 1 to finish..."
wait
echo ">>> Group 1 done."
echo ""

# --- Group 2: treatment (with retrieval) ---
echo ">>> [Group 2/2] ablation_with_retrieval (ExperienceJudge ON, retrieval ON)"
echo ""
for task in ${TASKS}; do
    TASK_DIR="${EXAMPLES_DIR}/${task}"
    CONFIG="${TASK_DIR}/config_ablation_with_retrieval.yaml"
    INIT="${TASK_DIR}/init.cpp"
    EVALUATOR="${TASK_DIR}/evaluator.py"

    echo "  Starting: ${task} (with_retrieval)..."
    python "${ROOT_DIR}/run_famou.py" \
        -c "${CONFIG}" \
        -p "${INIT}" \
        -e "${EVALUATOR}" &
    echo "    PID: $!"
done

echo ""
echo ">>> Waiting for Group 2 to finish..."
wait
echo ">>> Group 2 done."
echo ""
echo "============================================"
echo "  All 10 experiments finished."
echo "  Results in: famou_data/ale_bench/"
echo "============================================"
