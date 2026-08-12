"""
Circle packing evaluator for Famou 2.0 (n=26).

This evaluator follows the Famou evaluator interface:
- Input: Path to program file containing construct_packing() function
- Output: Dict with metrics and combined_score

The evaluator runs the program in a subprocess with timeout, validates the
packing solution (no overlaps, all circles inside unit square), and computes
a fitness score based on the sum of radii.

Usage with Famou:
    from famou.modules import EvaluateModule
    from examples.circle_packing.evaluator import evaluate

    evaluator = EvaluateModule(name="eval", evaluate_fn=evaluate)
    result = evaluator(context, rollout_result)

Target: AlphaEvolve benchmark result of 2.635 for n=26 circles
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle
import math


class TimeoutError(Exception):
    """
    TimeoutError
    """

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square.
    """
    if not isinstance(centers, np.ndarray) or not isinstance(radii, np.ndarray):
        print(f"Centers or radii are not numpy arrays. Got {type(centers)}, {type(radii)}")
        return False, f"Centers or radii are not numpy arrays. Got {type(centers)}, {type(radii)}"
        
    n = radii.shape[0]
    if n != 26 or centers.shape != (26, 2) or radii.shape != (26,):
        print(f"Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected (26, 2) and (26,)")
        return False, f"Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected (26, 2) and (26,)"
        
    for i in range(n):
        x, y = centers[i]; r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False, f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False, f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}"
    return True, ""


def run_with_timeout(program_path, timeout_seconds=20, gpu_ids=None):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds
        gpu_ids: Optional list of GPU IDs to use (e.g., [0] or [0, 1])

    Returns:
        centers, radii, sum_radii tuple from the program
    """
    # Print GPU allocation for visibility
    if gpu_ids is not None:
        print(f"Evaluate [GPU] Using GPU: {gpu_ids}")

    # Prepare environment with GPU configuration
    env = os.environ.copy()
    if gpu_ids is not None:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback
import inspect

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
# print(f"Running in subprocess, Python version: {{sys.version}}")
# print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the packing function
    # print("Calling run_packing()...")
    
    #run_packing = getattr(program, "run_packing", None) 
    construct_packing = getattr(program, "construct_packing", None)
    entry =  construct_packing
    if entry is None:
        raise AttributeError("new_born program.py has neither run_packing nor construct_packing function")

    centers, radii, sum_radii = entry()
    # print(f"run_packing() returned successfully: sum_radii = {{sum_radii}}")

    # Save results to a file
    results = {{
        'centers': centers,
        'radii': radii,
        'sum_radii': sum_radii
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    # print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    tb = traceback.format_exc()
    # 尽量把源函数也带回
    try:
        if entry is not None:
            fn = inspect.unwrap(entry)
            function = inspect.getsource(fn)
        else:
            function = ""
    except Exception:
        function = ""
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': f'{{e}}', 'traceback': tb, 'function': function}}, f)
    # 关键：即使失败也正常退出，避免父进程看到非 0 退出码
    sys.exit(0)
""" 
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout and configured environment
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env  # Pass environment with GPU configuration
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            #print(f"Subprocess stdout: {stdout.decode()}")
            #if stderr:
            #    print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                if "error" in results:
                    # 把子进程错误包装后往上返回/抛出（由上层决定是否继续）
                    err = results.get("error", "")
                    tb = results.get("traceback", "")
                    fn = results.get("function", "")
                    raise RuntimeError(f"Program execution failed: {err}\nTraceback:\n{tb}\nFunction:\n{fn}")
                return results["centers"], results["radii"], results["sum_radii"]

            # 如果没有结果文件，再根据退出码决定是否抛错，并附带 stdout/stderr
            if exit_code != 0:
                raise RuntimeError(
                    f"Process exited with code {exit_code}\n"
                    f"---- STDOUT ----\n{stdout.decode(errors='ignore')}\n"
                    f"---- STDERR ----\n{stderr.decode(errors='ignore')}"
                )
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(path_user_py, gpu_ids=None):
    """
    Evaluate a circle packing solution.

    Runs the program in a subprocess with timeout, validates the packing
    (circles don't overlap and fit inside unit square), and computes fitness
    based on sum of radii compared to target.

    Args:
        path_user_py: Path to the program file containing construct_packing() function
        gpu_ids: Optional list of GPU IDs allocated for this evaluation (e.g., [0] or [0, 1])

    Returns:
        Dictionary with evaluation results:
        - combined_score: Float (target_ratio * validity) - REQUIRED overall fitness
        - sum_radii: Total sum of all radii
        - target_ratio: sum_radii / target (2.635)
        - validity: 1.0 if valid packing, 0.0 otherwise
        - eval_time: Execution time in seconds
        - gpu_ids: List of GPU IDs used (for logging/monitoring)
        - error_info: Dict with error details if any

        Note: EvaluateModule will extract all fields except 'combined_score',
        'execution_status', and 'error_message' as metrics.

    Example:
        >>> result = evaluate("/path/to/solution.py", gpu_ids=[0])
        >>> result["combined_score"]
        0.95
        >>> result["sum_radii"]
        2.503
    """
    program_path = path_user_py
    # Target value from the paper
    TARGET_VALUE = 2.635  # AlphaEvolve result for n=26

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()
        # Use subprocess to run with timeout and GPU allocation
        centers, radii, reported_sum = run_with_timeout(
            program_path,
            timeout_seconds=600,  # Single timeout
            gpu_ids=gpu_ids  # Pass GPU allocation
        )
        end_time = time.time()
        eval_time = end_time - start_time

        # Ensure centers and radii are numpy arrays
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        # Validate solution
        valid, error_info = validate_packing(centers, radii)
        if math.isnan(reported_sum):
            return {
                "sum_radii": 0.0, "target_ratio": 0.0, "validity": 0.0, "eval_time": eval_time, "combined_score": 0.0,
                "error_info": "Reported sum is nan"
            }
        if reported_sum < 0 or any(x < 0 for x in radii):
            return {
                "sum_radii": 0.0, "target_ratio": 0.0, "validity": 0.0, "eval_time": eval_time, "combined_score": 0.0,
                "error_info": "Radii cannot be negative"
            }

        sum_radii = np.sum(radii) if valid else 0.0

        if valid and abs(sum_radii - reported_sum) > 1e-6:
            return {
                "sum_radii": 0.0, "target_ratio": 0.0, "validity": 0.0, "eval_time": eval_time, "combined_score": 0.0,
                "error_info": f"Reported sum {reported_sum} doesn't match calculated sum {sum_radii}"
            }

        # Target ratio (how close we are to the target)
        target_ratio = sum_radii / TARGET_VALUE if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Combined score - higher is better
        combined_score = target_ratio * validity

        return {
            "sum_radii": float(sum_radii),
            "target_ratio": float(target_ratio),
            "validity": float(validity),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
            "gpu_ids": gpu_ids,  # Include GPU info in results for tracking
            "error_info": error_info if error_info else None,
        }

    except Exception as e:
        return {
            "sum_radii": 0.0,
            "target_ratio": 0.0,
            "validity": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
            "error_info": str(e) if e else None,
        }


from pathlib import Path
import json
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python evaluator_lm.py path/to/your_program.py"
        )
        sys.exit(1)
    if ("json" in sys.argv[1]):
        json_path = Path(sys.argv[1])
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        code = data["code"]
        # code = "from scipy.optimize import differential_evolution, LinearConstraint\n" + code
        print(code)
        output_path = "temp.py"
        output = Path(output_path)
        with output.open("w", encoding="utf-8", newline="") as f:
            f.write(code)
    else:
        output_path = sys.argv[1]


    result = evaluate(output_path)
    print("---FINAL ERROR INFO---")
    print(result.get("error_info", "None"))
    print("--- RESULT ---")
    print(result)