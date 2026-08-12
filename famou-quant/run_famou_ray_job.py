#!/usr/bin/env python3
"""
Ray Job 通用提交脚本。

通过 Ray Job Submission SDK 将 famou 实验提交到远程 Ray 集群，
并实时轮询日志直到任务结束。收到 SIGINT/SIGTERM 时自动取消远程任务。

Usage (CLI)::

    # 单种子
    python run_famou_ray_job.py \\
        --config  examples/circle_packing/config_ray.yaml \\
        --program examples/circle_packing/init.py \\
        --evaluator examples/circle_packing/evaluator.py \\
        [--job-name my-experiment] \\
        [--ray-address http://10.94.147.104:8265]

    # 多种子(多个 -p,空格分隔;round-robin 分到各 island 作初始种群)
    python run_famou_ray_job.py \\
        -c exp/config_ray.yaml \\
        -p exp/init.py exp/init_le_svm.py exp/init_bnt.py \\
        -e exp/evaluator.py

Usage (Python)::

    from run_famou_ray_job import submit_ray_job
    submit_ray_job(
        config="config.yaml",
        program="init.py",                       # 单个 str
        # 或多个:program=["init.py", "init_le_svm.py", "init_bnt.py"]
        evaluator="evaluator.py",
    )

Environment Variables (optional, CLI args take precedence):
    RAY_ADDRESS : Ray dashboard HTTP address (default: http://127.0.0.1:8265)
    JOB_NAME    : Submission ID shown in Ray Dashboard (default: auto-generated)
"""

import argparse
import os
import re
import signal
import sys
import time
from datetime import datetime

from ray.job_submission import JobSubmissionClient, JobStatus


# =============================================================================
# 项目路径
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FAMOU_SRC = os.path.join(PROJECT_ROOT, "famou")
REQ_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")


def parse_requirements(req_file: str) -> dict:
    """Parse pip dependencies and source config from a requirements.txt file.

    Extracts three categories of information:
      - ``--index-url``  : custom PyPI index URL
      - ``--trusted-host``: trusted host for the index
      - package lines    : pure dependency specifiers (e.g. ``numpy>=1.24.0``)

    Lines starting with ``-`` (other than the two above) are skipped,
    including ``-e .``, ``--extra-index-url``, etc.

    Args:
        req_file: Absolute path to the requirements.txt file.

    Returns:
        A dict with keys:
          - ``pip_deps``     (list[str]): package specifiers
          - ``index_url``    (str | None): PyPI index URL
          - ``trusted_host`` (str | None): trusted host name
    """
    pip_deps = []
    index_url = None
    trusted_host = None

    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            m = re.match(r"^--index-url\s+(.+)$", line)
            if m:
                index_url = m.group(1).strip()
                continue

            m = re.match(r"^--trusted-host\s+(.+)$", line)
            if m:
                trusted_host = m.group(1).strip()
                continue

            if line.startswith("-"):
                continue

            pip_deps.append(line)

    return {
        "pip_deps": pip_deps,
        "index_url": index_url,
        "trusted_host": trusted_host,
    }


def submit_ray_job(
    config: str,
    program,
    evaluator: str,
    job_name: str = None,
    ray_address: str = None,
):
    """Submit a famou experiment as a Ray Job and stream logs until completion.

    The function performs the following steps:
      1. Build ``runtime_env`` from *requirements.txt* (pip deps, py_modules, env_vars).
      2. Submit the job via :class:`JobSubmissionClient`.
      3. Register SIGINT / SIGTERM handlers that call ``client.stop_job()``
         to cancel the remote job before exiting.
      4. Poll ``get_job_logs()`` every second, printing incremental output.
      5. Exit with code 0 on ``SUCCEEDED``, 1 otherwise.

    Args:
        config:      Path to the experiment YAML config file.
        program:     Path (str) or list of paths to the initial seed program
                     file(s). Multiple seeds enable multi-seed / multi-island
                     experiments (round-robin assigned to islands by the kernel).
        evaluator:   Path to the evaluator Python module.
        job_name:    Submission ID displayed in Ray Dashboard.
                     Falls back to ``$JOB_NAME`` env var, then auto-generates
                     ``famou-<config_stem>-<timestamp>``.
        ray_address: Ray dashboard HTTP address (e.g. ``http://host:8265``).
                     Falls back to ``$RAY_ADDRESS`` env var, then ``http://127.0.0.1:8265``.

    Raises:
        SystemExit: Always exits via ``sys.exit()`` — 0 on success, 1 on failure,
                    130 on signal interruption.
    """
    # ---- 参数处理 ----
    ray_address = ray_address or os.environ.get("RAY_ADDRESS", "http://127.0.0.1:8265")
    config = os.path.abspath(config)
    # program 可为单个路径(str)或多个路径(list/tuple),统一成绝对路径列表
    program_list = [program] if isinstance(program, str) else list(program)
    program_list = [os.path.abspath(p) for p in program_list]
    evaluator = os.path.abspath(evaluator)

    if not job_name:
        job_name = os.environ.get("JOB_NAME", "")
    if not job_name:
        config_stem = os.path.splitext(os.path.basename(config))[0]
        job_name = f"famou-{config_stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # ---- 解析 requirements.txt ----
    req_info = parse_requirements(REQ_FILE)

    env_vars = {"RAY_ADDRESS": "auto"}
    if req_info["index_url"]:
        env_vars["PIP_INDEX_URL"] = req_info["index_url"]
    if req_info["trusted_host"]:
        env_vars["PIP_TRUSTED_HOST"] = req_info["trusted_host"]

    runtime_env = {
        "py_modules": [FAMOU_SRC],
        "pip": req_info["pip_deps"],
        "env_vars": env_vars,
    }

    entrypoint = (
        f"python {PROJECT_ROOT}/run_famou.py"
        f" -c {config}"
        f" -p {' '.join(program_list)}"
        f" -e {evaluator}"
    )

    # ---- 打印信息 ----
    print("=== Ray Job Submit ===")
    print(f"  RAY_ADDRESS : {ray_address}")
    print(f"  JOB_NAME    : {job_name}")
    print(f"  FAMOU_SRC   : {FAMOU_SRC}")
    print(f"  CONFIG      : {config}")
    print(f"  PROGRAM(S)  : {program_list}")
    print(f"  EVALUATOR   : {evaluator}")
    print(f"  PIP_INDEX   : {req_info['index_url']}")
    print(f"  PIP_TRUSTED : {req_info['trusted_host']}")
    print(f"  PIP_DEPS    : {req_info['pip_deps']}")
    print()

    # ---- 提交任务 ----
    client = JobSubmissionClient(ray_address)
    submission_id = client.submit_job(
        entrypoint=entrypoint,
        submission_id=job_name,
        runtime_env=runtime_env,
    )

    print(f"Job '{submission_id}' 已提交，正在等待运行并实时输出日志...")
    print(f"  Dashboard : {ray_address}")
    print()

    # ---- 信号处理: 收到终止信号时取消远程任务 ----
    def _signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM by stopping the remote Ray Job."""
        sig_name = signal.Signals(signum).name
        print(f"\n收到 {sig_name}，正在取消 Ray Job '{submission_id}'...")
        try:
            client.stop_job(submission_id)
            print("已发送取消请求。")
        except Exception as e:
            print(f"取消请求失败: {e}")
        sys.exit(130)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ---- 实时跟踪日志 (轮询方式) ----
    prev_log_len = 0
    terminal_statuses = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}

    while True:
        status = client.get_job_status(submission_id)
        logs = client.get_job_logs(submission_id)
        if len(logs) > prev_log_len:
            print(logs[prev_log_len:], end="")
            prev_log_len = len(logs)
        if status in terminal_statuses:
            break
        time.sleep(1)

    # ---- 获取最终状态 ----
    print()
    print("=== Job 结束 ===")
    print(f"  JOB_NAME : {submission_id}")
    print(f"  STATUS   : {status}")

    if status == JobStatus.SUCCEEDED:
        sys.exit(0)
    else:
        sys.exit(1)


def main():
    """CLI entry point. Parse arguments and delegate to :func:`submit_ray_job`."""
    parser = argparse.ArgumentParser(description="Ray Job 通用提交脚本")
    parser.add_argument("--config", "-c", required=True, help="实验配置文件")
    parser.add_argument(
        "--programs", "--program", "-p",
        dest="programs",
        nargs="+",
        required=True,
        help="初始程序文件;可传多个(空格分隔)作多种子/多岛实验的初始种群",
    )
    parser.add_argument("--evaluator", "-e", required=True, help="评估器文件")
    parser.add_argument("--job-name", default=None, help="任务名称")
    parser.add_argument("--ray-address", default=None, help="Ray dashboard 地址")
    args = parser.parse_args()

    submit_ray_job(
        config=args.config,
        program=args.programs,
        evaluator=args.evaluator,
        job_name=args.job_name,
        ray_address=args.ray_address,
    )


if __name__ == "__main__":
    main()
