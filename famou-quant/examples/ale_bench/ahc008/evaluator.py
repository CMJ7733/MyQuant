import subprocess
from pathlib import Path
import time
import random
import tempfile
import re
import signal
# 移除了 concurrent.futures 导入
import os
import numpy as np
import logging
import uuid  # 用于生成唯一的临时文件名

# 超时后清理 communicate 的等待时间（秒）
_CLEANUP_TIMEOUT = 15


def _kill_process_tree(proc):
    """Kill subprocess and its entire process tree via process group.

    When Popen is created with preexec_fn=os.setsid, the child becomes
    a session leader with its own process group.  os.killpg sends SIGKILL
    to every process in that group, including grandchildren that proc.kill()
    would miss.
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # Process already dead or can't get pgid — fall back to proc.kill()
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


def _safe_communicate(proc, timeout=_CLEANUP_TIMEOUT):
    """Call proc.communicate() with a timeout to avoid hanging forever."""
    try:
        proc.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError, Exception):
        # Last resort: close pipes manually so we don't block
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe:
                try:
                    pipe.close()
                except Exception:
                    pass


def run_algorithm(input_file: Path, code_file: Path, language: str, timeout: int):
    """
    运行指定的算法代码，并将输入从文件重定向，输出到临时文件。
    支持 Python 和 C++ 语言。函数返回的是整个函数的运行时间。

    Args:
        input_file (Path): 包含算法输入数据的 Path 对象。
        code_file (Path): 包含算法代码的 Path 对象。
        language (str): 算法的编程语言 ('Python' 或 'C++')。

    Returns:
        tuple[Path | None, float, str | None]:
            - output_path (Path): 存储算法输出的临时文件路径，如果失败则为 None。
            - execution_time (float): 整个函数的总运行时间（秒）。
            - error_message (str): 错误信息，如果成功则为 None。
    """
    start_at_total = time.perf_counter()
    output_path: Path | None = None
    temp_executable_path: Path | None = None  # 用于C++编译的临时可执行文件路径

    # 1. 检查输入和代码文件是否存在
    if not input_file.exists():
        return None, (time.perf_counter() - start_at_total), f"输入文件不存在: {input_file}"
    if not code_file.exists():
        return None, (time.perf_counter() - start_at_total), f"算法代码文件不存在: {code_file}"

    proc = None
    try:
        # 2. 创建一个安全的临时文件来存储输出
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".txt", encoding="utf-8"
        ) as temp_output_file:
            output_path = Path(temp_output_file.name)

        # 3. 根据语言构建命令
        run_command_parts: list[str] = []  # 实际运行算法的命令及参数

        if language == "Python":
            run_command_parts = ["python", str(code_file)]

        elif language == "C++17":
            # --- C++ 编译阶段 ---
            # 生成一个唯一的临时可执行文件路径
            temp_executable_name = f"temp_exec_{uuid.uuid4().hex}"
            temp_executable_path = Path(tempfile.gettempdir()) / temp_executable_name

            compile_command_parts = [
                "g++",
                str(code_file),
                "-o",
                str(temp_executable_path),
                "-lboost_system",
                "-lboost_filesystem",
                "-lboost_thread",
                "-lboost_program_options",
                "-lboost_date_time",
                "-lboost_regex",
                "-lgmp",
                "-lgmpxx",
            ]

            try:
                # 执行编译命令
                compile_proc = subprocess.run(
                    compile_command_parts,
                    stdout=subprocess.PIPE,  # 捕获编译器的标准输出
                    stderr=subprocess.PIPE,  # 捕获编译器的标准错误
                    text=True,  # 自动解码输出为文本
                )

                if compile_proc.returncode != 0:
                    # 编译失败
                    error_message = (
                        f"C++ compilation Error (Exit Code: {compile_proc.returncode})。\n"
                        f"compile stdout:\n{compile_proc.stdout}\n"
                        f"compile stderr:\n{compile_proc.stderr}"
                    )
                    return None, (time.perf_counter() - start_at_total), error_message

            except FileNotFoundError:
                error_message = "编译失败：找不到 'g++' 命令，请检查 C++ 编译器是否已安装并配置在 PATH 中。"
                return None, (time.perf_counter() - start_at_total), error_message

            run_command_parts = [str(temp_executable_path)]

        else:
            # 不支持的语言
            error_message = f"编译错误"
            return None, (time.perf_counter() - start_at_total), error_message

        start_at_exec = time.perf_counter()

        # --- 算法执行阶段 (Python 和 C++ 共享) ---
        with open(input_file, "r", encoding="utf-8") as stdin_handle:
            with open(output_path, "w", encoding="utf-8") as stdout_handle:
                proc = subprocess.Popen(
                    run_command_parts,
                    stdin=stdin_handle,  # 将输入文件句柄作为标准输入
                    stdout=stdout_handle,  # 将输出文件句柄作为标准输出
                    stderr=subprocess.PIPE,  # 捕获标准错误输出
                    text=True,  # 自动解码 stderr
                    encoding="utf-8",  # 建议在 Popen 中也指定 encoding
                    preexec_fn=os.setsid,  # 创建新进程组，便于杀整个进程树
                )
                # 注意：proc.communicate() 会等待进程结束
                _, stderr_output = proc.communicate(timeout=2 * timeout)



        # 算法执行结束，计算总时间
        execution_time = time.perf_counter() - start_at_exec

        # 检查子进程的返回码
        if proc.returncode != 0:
            error_message = (
                f"Algorithm Execution Error (Exit Code: {proc.returncode})。\n"
                f"standard error output:\n{stderr_output}"
            )
            if output_path.exists():
                output_path.unlink()
            return None, execution_time, error_message

        # 成功执行
        return output_path, execution_time, None

    except FileNotFoundError:  # 这通常发生在运行阶段找不到可执行文件
        error_message = f"执行失败：找不到 '{run_command_parts[0]}' 命令或编译后的可执行文件，请检查环境或 PATH。"
        if output_path and output_path.exists():
            output_path.unlink()

        return None, (time.perf_counter() - start_at_total), error_message

    except subprocess.TimeoutExpired:
        # proc.communicate() 超时会引发此异常
        if proc:
            _kill_process_tree(proc)
            _safe_communicate(proc)
        error_message = "Algorithm Execution Timeout."
        return None, (time.perf_counter() - start_at_total), error_message

    except Exception as e:
        # 捕获其他所有意外错误
        error_message = f"Run error: {e}"
        if output_path and output_path.exists():
            output_path.unlink()
        return None, (time.perf_counter() - start_at_total), error_message

    finally:
        # 确保进程被终止
        if proc and proc.poll() is None:
            _kill_process_tree(proc)
            _safe_communicate(proc)

        # --- 清理阶段 ---
        # 确保删除 C++ 编译生成的临时可执行文件
        if temp_executable_path and temp_executable_path.exists():
            try:
                temp_executable_path.unlink()
            except OSError as e:
                # 权限问题或文件正在被使用可能导致无法删除
                print(f"警告: 无法删除临时可执行文件 {temp_executable_path}: {e}")


def run_algorithm_reactive(
    input_file: Path, code_file: Path, evaluator_file: Path, language: str, timeout: int
):
    """
    运行算法评估脚本并返回分数、运行时间和错误信息。
    """

    tester_path = str(evaluator_file)

    input_file_handle = None
    output_file_handle = None
    output_temp_file_path_str = None

    temp_executable_path = None  # 用于 C++ 编译后的可执行文件路径

    score = None
    run_time = None
    run_error = None

    proc = None
    try:
        output_temp_file_obj = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".out", encoding="utf-8"
        )
        output_temp_file_path_str = output_temp_file_obj.name  # 存储路径字符串用于打印和清理
        output_file_handle = output_temp_file_obj  # 这才是传递给 stdout 的文件对象

        input_file_handle = open(input_file, "r", encoding="utf-8")

        executable_to_run: Path

        # --- 编译步骤 (仅针对 C++) ---
        if "c++" in language.lower():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as temp_exec_file:
                temp_executable_path = Path(temp_exec_file.name)

            compile_command_parts = [
                "g++",
                str(code_file),
                "-o",
                str(temp_executable_path),
                "-I",
                "/usr/include/eigen3",
                "-lboost_system",
                "-lboost_filesystem",
                "-lboost_thread",
                "-lboost_program_options",
                "-lboost_date_time",
                "-lboost_regex",
                "-lgmp",
                "-lgmpxx",
            ]

            compile_result = subprocess.run(
                compile_command_parts, check=False, capture_output=True, text=True
            )

            if compile_result.returncode != 0:
                run_error = (
                    f"C++ compilation failed with exit code {compile_result.returncode}.\n"
                    f"Stdout:\n{compile_result.stdout}\n"
                    f"Stderr:\n{compile_result.stderr}"
                )
                if temp_executable_path and os.path.exists(temp_executable_path):
                    os.remove(temp_executable_path)
                return score, run_time, run_error

            executable_to_run = temp_executable_path

        elif language.lower() == "python":
            executable_to_run = code_file
        else:
            raise ValueError(
                f"Unsupported language: {language}. Please use 'Python' or 'C++'."
            )

        command_args_for_tester = []
        if language.lower() == "python":
            command_args_for_tester.append("python")
            command_args_for_tester.append(str(executable_to_run))
        elif "c++" in language.lower():
            command_args_for_tester.append(str(executable_to_run))

        command_to_run = [tester_path] + command_args_for_tester

        start_at = time.perf_counter()

        proc = subprocess.Popen(
            command_to_run,
            stdin=input_file_handle,
            stdout=output_file_handle,
            stderr=subprocess.PIPE,
            text=True,  # 自动解码 stderr
            encoding="utf-8",  # 建议在 Popen 中也指定 encoding
            preexec_fn=os.setsid,  # 创建新进程组，便于杀整个进程树
        )
        _, stderr_output = proc.communicate(timeout=3 * timeout)

        run_time = time.perf_counter() - start_at

        if proc.returncode != 0:
            run_error = (
                f"Command failed with exit code {proc.returncode}.\n"
                f"Stderr:\n{stderr_output}"
            )

        pattern = r"Score\s*[:=]\s*(\d+(\.\d+)?)"
        match = re.search(pattern, stderr_output, re.IGNORECASE)
        if match and "Out of range:" not in stderr_output:
            score = float(match.group(1))
        else:
            run_error = (
                f"No score found in stderr output:\n{stderr_output}"
            )
            score = 0.0


    except subprocess.CalledProcessError as e:
        run_error = (
            f"Command failed with exit code {e.returncode}.\n"
            f"Stderr:\n{e.stderr}"
        )
    except subprocess.TimeoutExpired:
        if proc:
            _kill_process_tree(proc)
            _safe_communicate(proc)
        run_error = "Command timed out."
    except FileNotFoundError as e:
        run_error = (
            f"Executable not found. Check paths: "
            f"{tester_path} or interpreter/compiler for "
            f"{language}. Error: {e}"
        )
    except ValueError as e:
        run_error = f"Configuration error: {e}"
    except Exception as e:
        raise e
        run_error = f"An unexpected error occurred: {e}"
    finally:
        if proc and proc.poll() is None:
            _kill_process_tree(proc)
            _safe_communicate(proc)

        if input_file_handle:
            input_file_handle.close()
        if output_file_handle:  # NamedTemporaryFile 返回的对象有 close 方法
            output_file_handle.close()
            # 只有当 output_temp_file_path_str 被成功赋值且文件存在时才尝试删除
            if output_temp_file_path_str and os.path.exists(output_temp_file_path_str):
                os.remove(output_temp_file_path_str)  # 删除临时输出文件

        if temp_executable_path and os.path.exists(temp_executable_path):
            os.remove(temp_executable_path)

    return score, run_time, run_error


def judge_score(input_file, output_file, evaluator_file):
    """
    使用外部评估器对输入文件进行评估，并返回评估结果和评估时间。
    """
    command = [str(evaluator_file), str(input_file), str(output_file)]
    proc = None  # *** 修复：初始化 proc 变量 ***
    try:
        start_at = time.perf_counter()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE, # 捕获 stdout
            stderr=subprocess.PIPE, # 捕获 stderr
            text=True,
            encoding="utf-8",
            preexec_fn=os.setsid,  # 创建新进程组，便于杀整个进程树
        )
        stdout_output, stderr_output = proc.communicate(timeout=100)

        execution_time = time.perf_counter() - start_at

        if proc.returncode != 0:
            error_message = f"Evaluation error (exit code {proc.returncode}):\n{stderr_output}"
            return None, 0.0, error_message

        pattern = r"Score\s*[:=]\s*(\d+(\.\d+)?)"
        # 检查 stderr 和 stdout
        match = re.search(pattern, stderr_output, re.IGNORECASE) or \
                re.search(pattern, stdout_output, re.IGNORECASE)

        score = float(match.group(1)) if match else 0.0
        return score, execution_time, None

    except subprocess.CalledProcessError as e:
        error_message = f"Evaluation error (exit code {e.returncode}):\n{e.stderr}"
        return None, 0.0, error_message
    except subprocess.TimeoutExpired:
        if proc:
            _kill_process_tree(proc)
            _safe_communicate(proc)
        error_message = "Evaluation timed out."
        return None, 0.0, error_message
    except ValueError:
        # 如果评测程序的输出不是一个有效的浮点数
        error_message = f"Evaluation error: wrong format"
        return None, 0.0, error_message
    except Exception as e:
        error_message = f"Evaluation error: {e}"
        return None, 0.0, error_message
    finally:
        # *** 修复：添加 'if proc:' 检查 ***
        if proc and proc.poll() is None:
            _kill_process_tree(proc)
            _safe_communicate(proc)


def evaluate_one(
    code_file,
    input_file,
    evaluator_file,
    language="Python",
    timeout=10,
    score_type="maximize",
    problem_type="batch",
):
    """
    评估单个代码文件。
    """
    final_result = {
        "validity": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
        "run_time": 0.0,
        "evaluate_time": 0.0,
        "error_info": {},
    }

    user_output_file = None
    try:
        if problem_type == "batch":
            # 步骤 1: 运行用户算法
            user_output_file, run_time, run_error = run_algorithm(
                input_file, code_file, language, timeout
            )

            if run_error:
                final_result["error_info"]["run_error"] = f"Solution runtime error: {run_error}"
                if score_type == "minimize":
                    final_result["score"] = float('inf')
                return final_result  # 执行失败，提前返回

            # 步骤 2: 评测打分
            score, evaluate_time, judge_error = judge_score(
                input_file, user_output_file, evaluator_file
            )

            if judge_error:
                final_result["error_info"][
                    "evaluator_error"
                ] = f"Solution evaluation error: {judge_error}"
                final_result["validity"] = 0.0
                return final_result

            # 如果一切顺利
            final_result["score"] = (
                score
            )
            final_result["run_time"] = run_time
            final_result["evaluate_time"] = evaluate_time
            final_result["validity"] = 1.0
            return final_result
        else:
            score, run_time, run_error = run_algorithm_reactive(
                input_file, code_file, evaluator_file, language, timeout
            )

            if run_error:
                final_result["error_info"]["run_error"] = f"Solution runtime error: {run_error}"
                if score_type == "minimize":
                    final_result["score"] = float('inf')
                return final_result

            final_result["score"] = (
                score
            )
            final_result["run_time"] = run_time
            final_result["evaluate_time"] = 0
            final_result["validity"] = 1.0
            return final_result

    finally:
        # 步骤 3: 清理资源
        # 无论成功或失败，都尝试删除用户算法生成的临时文件
        if user_output_file and user_output_file.exists():
            user_output_file.unlink()


def evaluate(
    path_user_py,
    input_dir="/mnt/cfs_bj_mt/workspace/lihuaiming/ALE-Bench/ahc008/tools/in",
    evaluator_file="/mnt/cfs_bj_mt/workspace/lihuaiming/ALE-Bench/ahc008/tools/target/debug/tester",
    language="C++17",
    timeout=3.0,
    score_type="maximize",
    problem_type="reactive",
    target=7764801.6600
):
    """
    评估用户代码文件（串行执行）。
    """

    input_dir = Path(input_dir)
    code_file = Path(path_user_py)
    evaluator_file = Path(evaluator_file)

    final_result = {
        "validity": 0.0,
        "combined_score": 0.0,
        "score_list": [],
        "score": 0.0,
        "total_run_time": 0.0,
        "avg_run_time": 0.0,
        "max_run_time": 0.0,
        "error_info": {},
    }
    file_num = 0

    # 过滤出所有文件
    input_files_to_process = [f for f in input_dir.iterdir() if f.is_file()]

    if not input_files_to_process:
        print(f"No files found in {input_dir}")
        final_result["error_info"] = str(final_result["error_info"])
        return final_result

    # --- 移除 ThreadPoolExecutor ---
    # 使用简单的 for 循环串行执行
    for input_file in input_files_to_process:
        try:
            # 直接调用 evaluate_one
            result = evaluate_one(
                code_file,
                input_file,
                evaluator_file,
                language,
                timeout,
                score_type,
                problem_type,
            )
        except Exception as exc:
            # 捕获 evaluate_one 内部未捕获的异常
            print(f"'{input_file.name}' generated an exception: {exc}")
            final_result['validity'] = 0
            final_result['error_info'] = {"evaluation_exception": str(exc)}
            final_result["error_info"] = str(final_result["error_info"])
            return final_result # 发生严重错误，终止整个评估
        else:
            # 聚合结果
            for k, v in result["error_info"].items():
                if v not in final_result["error_info"].get(k, ""):
                    final_result["error_info"][k] = final_result["error_info"].get(k, "") + "\n" + v
                # 如果有编译错误，立即停止并返回
                if "compilation" in v or "compile" in v:
                    result["error_info"] = str(result["error_info"])
                    return result

            final_result['validity'] += result["validity"]
            final_result["score"] += result["score"]
            final_result["total_run_time"] += result["run_time"]
            final_result["max_run_time"] = max(final_result["max_run_time"], result["run_time"])
            final_result['score_list'].append(result["score"])

            file_num += 1

    # --- 评估循环结束 ---

    if file_num > 0:
        final_result["avg_run_time"] = final_result["total_run_time"] / file_num
        final_result["score"] /= file_num
        final_result['validity'] /= file_num
    else:
        final_result["avg_run_time"] = 0.0  # 避免除零错误

    if final_result["validity"] != 1 or final_result["score"] == 0:
        final_result["combined_score"] = 0
    else:
        if target is not None:
            if score_type == "maximize":
                final_result["combined_score"] = final_result["score"] / target
            else:
                final_result["combined_score"] = 1 - (final_result["score"] - target) / final_result["score"]
        else:
            if score_type == "maximize":
                final_result["combined_score"] = final_result["score"]
            else:
                final_result["combined_score"] = float(10000000 /  np.log10(100 + final_result["score"]))


    final_result["error_info"] = str(final_result["error_info"])
    return final_result


if __name__ == "__main__":
    result = evaluate(
        "/mnt/cfs_bj_mt/workspace/lihuaiming/fm_output_metric/gemini_api_ahc008_cluster_20260306_055931_ccdd/programs/131_5_0ae05d3b.cpp",
        input_dir="/mnt/cfs_bj_mt/workspace/wuchufan/ALE-Bench/ahc008/tools/in",
        evaluator_file="/mnt/cfs_bj_mt/workspace/wuchufan/ALE-Bench/ahc008/tools/target/debug/tester",
        language="C++",
        timeout=3,
        score_type="maximize",
        problem_type="reactive",
    )
    print(result)
