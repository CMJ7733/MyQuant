#!/usr/bin/env python3
"""
CVRP ALNS Evaluator for Famou 2.0 (Planning-based Evolution).

This evaluator follows the Famou 2.0 evaluator interface:
- Input: Path to program file (alns_operators.h)
- Output: Dict with metrics and combined_score

The evaluator:
1. Creates a temporary workspace with UUID naming (concurrent-safe)
2. Copies src template + evolved alns_operators.h
3. Compiles using g++
4. Runs batch_solver on 128 CVRP instances
5. Parses output and computes combined_score
6. Optionally preserves temp directory for debugging

Score formula: combined_score = max(0, 100 - (avg_final_cost - 36.6))
"""

import os
import sys
import subprocess
import shutil
import re
import uuid
from pathlib import Path
from typing import Dict

# Path configuration
CVRP_DIR = Path(__file__).parent.resolve()
ALNS_ENGINE_DIR = CVRP_DIR / "alns_engine"
SRC_TEMPLATE = ALNS_ENGINE_DIR / "src"
DATA_TXT = CVRP_DIR / "data" / "vrp500_test_luo.txt"

# Score calculation parameters
BASELINE_COST = 39.10  # Baseline cost (reference)
BEST_KNOWN_COST = 36.6  # Best known cost
MAX_SCORE = 100.0  # Maximum score

# Whether to keep temp directory for debugging (default: True)
# Set CVRP_KEEP_TEMP=0 to auto-delete after evaluation
KEEP_TEMP_DIR = os.environ.get("CVRP_KEEP_TEMP", "1") == "1"


def create_temp_workspace(evolved_operators_path: str) -> Path:
    """
    Create temporary workspace (UUID-named, concurrent-safe).

    Args:
        evolved_operators_path: Path to evolved alns_operators.h

    Returns:
        Temporary directory path
    """
    unique_id = uuid.uuid4().hex[:12]
    temp_base = CVRP_DIR / "tmp"
    temp_base.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_base / f"alns_eval_{unique_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Copy entire src directory structure
    temp_src = temp_dir / "src"
    try:
        shutil.copytree(SRC_TEMPLATE, temp_src, dirs_exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to copy src template: {e}")

    # Copy evolved alns_operators.h (overwrite placeholder)
    if not os.path.exists(evolved_operators_path):
        raise FileNotFoundError(f"Evolved file not found: {evolved_operators_path}")

    target_operators = temp_src / "alns_operators.h"
    shutil.copy(evolved_operators_path, target_operators)

    return temp_dir


def compile_in_temp_dir(temp_dir: Path) -> Path:
    """
    Compile in temporary directory using g++.

    Args:
        temp_dir: Temporary workspace directory

    Returns:
        Path to compiled executable

    Raises:
        RuntimeError: Compilation failed
    """
    temp_src = temp_dir / "src"
    binary_path = temp_dir / "batch_solver"

    compile_cmd = [
        "g++",
        "-std=c++17",
        "-O3",
        "-march=native",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        f"-I{temp_src}",
        "-o", str(binary_path),
        str(temp_src / "batch_solver.cpp"),
        str(temp_src / "core" / "alns_framework.cpp"),
        "-lpthread"
    ]

    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Compilation timeout after 60s")

    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else result.stdout
        raise RuntimeError(f"Compilation failed:\n{error_msg}")

    if not binary_path.exists():
        raise RuntimeError(f"Binary not found after compilation: {binary_path}")

    return binary_path


def run_batch_evaluation(
    binary_path: Path,
    temp_dir: Path,
    num_workers: int = 10,
    time_limit: float = 60.0,
    num_instances: int = 128,
) -> Dict[str, float]:
    """
    Run batch evaluation.

    Args:
        binary_path: Path to compiled executable
        temp_dir: Temporary directory (for saving CSV)
        num_workers: Number of concurrent workers
        time_limit: Time limit per instance (seconds)
        num_instances: Number of instances to evaluate (default: 128, max in data)

    Returns:
        Dictionary of evaluation metrics

    Raises:
        RuntimeError: Evaluation failed
        TimeoutError: Evaluation timeout
        FileNotFoundError: Data file not found
    """
    if not DATA_TXT.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_TXT}\n"
            f"Please run: python convert_pkl_to_txt.py"
        )

    csv_output = temp_dir / "results.csv"

    cmd = [
        str(binary_path),
        "--data", str(DATA_TXT),
        "--workers", str(num_workers),
        "--time", str(time_limit),
        "--seed", "42",
        "--count", str(num_instances),
        "--output", str(csv_output)
    ]

    # Total timeout: time_limit * num_instances / num_workers + 120s buffer
    total_timeout = int((time_limit * num_instances / num_workers) + 120)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=total_timeout
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Batch evaluation timeout after {total_timeout}s")

    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else result.stdout
        raise RuntimeError(f"Batch solver failed (exit code {result.returncode}):\n{error_msg}")

    output = result.stdout
    metrics = parse_output(output)

    return metrics


def parse_output(output: str) -> Dict[str, float]:
    """
    Parse batch_solver output.

    Args:
        output: stdout string

    Returns:
        Metrics dictionary

    Raises:
        ValueError: Cannot parse key metrics
    """
    avg_final_cost = None
    avg_improvement_pct = None
    feasible_rate = None

    for line in output.split('\n'):
        if "Final cost (avg):" in line:
            match = re.search(r'Final cost \(avg\):\s+([\d.]+)', line)
            if match:
                avg_final_cost = float(match.group(1))

        elif "Improvement (avg):" in line:
            match = re.search(r'Improvement \(avg\):\s+([\d.]+)', line)
            if match:
                avg_improvement_pct = float(match.group(1))

        elif "Feasible rate:" in line:
            match = re.search(r'Feasible rate:.*?\(([\d.]+)%\)', line)
            if match:
                feasible_rate = float(match.group(1)) / 100.0

    if avg_final_cost is None:
        raise ValueError(f"Cannot parse avg_final_cost from output:\n{output}")

    return {
        "avg_final_cost": avg_final_cost,
        "avg_improvement_pct": avg_improvement_pct or 0.0,
        "feasible_rate": feasible_rate if feasible_rate is not None else 0.0,
        "baseline": BASELINE_COST,
        "best_known": BEST_KNOWN_COST,
        "improvement": BASELINE_COST - avg_final_cost,
    }


def cleanup_temp_dir(temp_dir: Path) -> None:
    """
    Clean up temporary directory.

    Args:
        temp_dir: Directory to delete
    """
    if KEEP_TEMP_DIR:
        return

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


def evaluate(path_user_py: str) -> Dict[str, float]:
    """
    Main evaluator interface for Famou 2.0.

    Workflow:
    1. Create UUID-named temporary directory
    2. Copy src template + evolved alns_operators.h
    3. Compile with g++
    4. Run batch_solver
    5. Parse output and return metrics
    6. Optional: preserve temp directory for debugging

    Args:
        path_user_py: Path to evolved alns_operators.h file

    Returns:
        {
            "combined_score": float,      # Main metric (higher is better)
            "avg_final_cost": float,      # Average final cost
            "improvement": float,         # Improvement relative to baseline
            "validity": float,            # Compile+run success (0 or 1)
            "error_info": dict            # Error info (if any)
        }
    """
    temp_dir = None

    try:
        # Step 1: Create temporary workspace
        temp_dir = create_temp_workspace(path_user_py)

        # Step 2: Compile
        binary_path = compile_in_temp_dir(temp_dir)

        # Step 3: Run evaluation (full: 128 instances, 60s per instance)
        metrics = run_batch_evaluation(
            binary_path,
            temp_dir,
            num_workers=4,
            time_limit=10.0,
            num_instances=2,
        )

        # Step 4: Calculate combined_score
        avg_final_cost = metrics["avg_final_cost"]
        feasible_rate = metrics["feasible_rate"]

        # Check feasibility
        if feasible_rate < 0.99:
            combined_score = 0.0
        else:
            # Score: 100 - (cost - best_known)
            combined_score = max(0.0, MAX_SCORE - (avg_final_cost - BEST_KNOWN_COST))

        result = {
            "combined_score": combined_score,
            "avg_final_cost": avg_final_cost,
            "avg_improvement_pct": metrics["avg_improvement_pct"],
            "baseline": metrics["baseline"],
            "best_known": metrics["best_known"],
            "improvement": metrics["improvement"],
            "feasible_rate": feasible_rate,
            "validity": 1.0,
        }

        return result

    except subprocess.TimeoutExpired as e:
        return {
            "combined_score": 0.0,
            "avg_final_cost": 999999.0,
            "validity": 0.0,
            "error_info": {"error": "Evaluation timeout", "details": str(e)}
        }

    except RuntimeError as e:
        return {
            "combined_score": 0.0,
            "avg_final_cost": 999999.0,
            "validity": 0.0,
            "error_info": {"error": "Compilation or runtime error", "details": str(e)}
        }

    except Exception as e:
        import traceback
        return {
            "combined_score": 0.0,
            "avg_final_cost": 999999.0,
            "validity": 0.0,
            "error_info": {"error": "Unexpected error", "details": str(e), "traceback": traceback.format_exc()}
        }

    finally:
        if temp_dir is not None:
            cleanup_temp_dir(temp_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluator.py <path_to_alns_operators.h>")
        print("\nExample:")
        print("  python evaluator.py alns_operators_initial.h")
        sys.exit(1)

    result = evaluate(sys.argv[1])

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)
    for key, value in sorted(result.items()):
        print(f"{key:25s}: {value}")
    print("=" * 60)
