"""
Test script for Famou 2.0 system.

Tests basic functionality of:
- RolloutEngine with module pipeline
- Evolver with full evolution loop

Uses mock LLM and mock evaluator for deterministic testing.
"""

import random
import tempfile
from typing import Any, Dict, List

from famou.infrastructure.llm.mock_client import MockLLMClient
from famou.infrastructure.env import LocalEnv
from famou.infrastructure.storage import LocalStorage
from famou.infrastructure.logger import LocalLogger
from famou.config import ExperimentConfig, IslandConfig, ModulesConfig
from famou.core.data import Program, Context, Experiment
from famou.core.protocol import Rollout
from famou.modules.select.random import RandomSelect
from famou.modules.generate.mutation import MutationGenerate
from famou.modules.evaluate import EvaluateModule
from famou.controller import RolloutEngine, Evolver
from famou.strategies import StrategyRegistry
from famou.utils import generate_experiment_id

# Rebuild Context model to resolve forward references
Context.model_rebuild()


# =============================================================================
# Mock Components
# =============================================================================

def mock_evaluator(program_path: str) -> Dict[str, Any]:
    """
    Mock evaluator that returns random scores.

    Args:
        program_path: Path to program file (not used in mock)

    Returns:
        Dict with combined_score, validity, and test metrics
    """
    return {
        "combined_score": 0.5 + random.uniform(0, 0.4),  # Random score 0.5-0.9
        "validity": 1.0,
        "test_metric": "mock_value",
        "random_metric": random.random(),
    }


def create_mock_llm() -> MockLLMClient:
    """Create mock LLM that returns valid Python code."""
    mock_code = '''
def construct_packing():
    """Mock implementation of circle packing."""
    import numpy as np

    # Simple mock - place circles in a grid pattern
    n = 26
    centers = np.zeros((n, 2))

    # Grid placement
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)

    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            centers[idx] = [(i + 1) * spacing, (j + 1) * spacing]
            idx += 1

    # Small uniform radii to avoid overlap
    radii = np.ones(n) * 0.03
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii
'''
    return MockLLMClient(response=f"```python\n{mock_code}\n```")


def create_seed_program() -> Program:
    """Create a seed program for testing."""
    seed_code = '''
def construct_packing():
    """Initial circle packing implementation."""
    import numpy as np

    n = 26
    centers = np.zeros((n, 2))

    # Place circles in concentric rings
    centers[0] = [0.5, 0.5]

    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    for i in range(17):
        angle = 2 * np.pi * i / 17
        centers[i + 9] = [0.5 + 0.45 * np.cos(angle), 0.5 + 0.45 * np.sin(angle)]

    centers = np.clip(centers, 0.05, 0.95)
    radii = np.ones(n) * 0.04

    return centers, radii, np.sum(radii)
'''
    return Program(
        id="seed_0",
        code=seed_code,
        generation=0,
        iteration=0,
        language="python",
        combined_score=0.5,  # Initial score
        validity=1.0,
    )


# =============================================================================
# Test 1: RolloutEngine
# =============================================================================

def test_rollout_engine():
    """
    Test RolloutEngine with a single rollout.

    Verifies:
    - Rollout pipeline executes correctly
    - Program is generated
    - Program has score and metrics
    """
    print("\n" + "=" * 70)
    print("TEST 1: RolloutEngine - Single Rollout")
    print("=" * 70)

    # Setup
    mock_llm = create_mock_llm()
    env = LocalEnv()
    logger = LocalLogger(
        name="test_engine",
        level="INFO",
        file_enabled=False,
        jsonl_enabled=False,
    )

    # Create temporary storage directory
    temp_dir = tempfile.mkdtemp(prefix="famou_test_")
    data_service = LocalStorage(base_path=temp_dir)

    # Create seed program
    seed_program = create_seed_program()

    # Create context
    context = Context(
        experiment_id="test_exp_001",
        task_description="Test circle packing optimization",
        population={"population": [seed_program]},
        island_id=0,
        config={"language": "python"},
    )

    # Create rollout pipeline
    rollout = Rollout([
        RandomSelect(),
        MutationGenerate(),
        EvaluateModule(evaluate_fn=mock_evaluator),
    ])

    # Create engine
    engine = RolloutEngine(
        llm_client=mock_llm,
        env=env,
        logger=logger,
        retry_enabled=False,  # Disable retry for testing
    )

    # Run rollout
    print("\n-> Running single rollout...")
    result = engine.execute_rollout(rollout, context, iteration=0)

    # Verify results
    assert result is not None, "Result should not be None"
    assert result.generated_program is not None, "Generated program should not be None"
    assert result.generated_program.combined_score is not None, "Program should have a score"
    assert result.generated_program.validity is not None, "Program should have validity"
    assert result.selected_parent_id is not None, "Should have selected parent"

    print(f"\n[OK] Rollout completed successfully")
    print(f"    - Selected parent: {result.selected_parent_id}")
    print(f"    - Generated program: {result.generated_program.id}")
    print(f"    - Score: {result.generated_program.combined_score:.4f}")
    print(f"    - Validity: {result.generated_program.validity}")
    print(f"    - Metrics: {result.generated_program.metrics}")

    return True


# =============================================================================
# Test 2: Full Evolver Loop
# =============================================================================

def test_evolver_loop():
    """
    Test Evolver with full evolution loop (3 iterations).

    Verifies:
    - Evolver runs multiple iterations
    - Population is updated after each iteration
    - Archive grows with new programs
    - Best score is tracked
    """
    print("\n" + "=" * 70)
    print("TEST 2: Evolver - Full Evolution Loop (3 iterations)")
    print("=" * 70)

    # Setup
    mock_llm = create_mock_llm()
    env = LocalEnv()

    # Create temporary storage directory
    temp_dir = "/Users/wuchufan/workspace/famou_v2/famou_test"
    experiment_id = generate_experiment_id("test")

    # Create logger with file output enabled (tests both console and file logging)
    logger = LocalLogger(
        name="test_evolver",
        experiment_id=experiment_id,
        log_dir=f"{temp_dir}/{experiment_id}",
        level="INFO",
        file_enabled=True,
        jsonl_enabled=True,
    )

    data_service = LocalStorage(base_path=temp_dir)

    # Create seed program
    seed_program = create_seed_program()

    # Create experiment
    experiment = Experiment(
        id=experiment_id,
        name="test_evolution",
        island_populations={0: {"population": [seed_program]}},
        archive=[seed_program],
        current_iteration=0,
    )

    # Create config
    config = ExperimentConfig(
        name="test_evolution",
        max_iterations=20,
        task_description="Test circle packing optimization",
        strategy="standard",  # Use strategy system
        island=IslandConfig(
            island_size=10,
        ),
        max_workers=1,  # Sequential for testing
        checkpoint_interval=100,  # High value to effectively disable during short test
    )

    # Create module parameters (optional, for testing config override)
    modules_config = ModulesConfig(
        select={"k": 1, "num_inspirations": 2},
        population={"max_size": 10},
    )

    # Load strategy from registry
    print("\n-> Loading strategy 'standard'...")
    strategy = StrategyRegistry.get(
        "standard",
        evaluate_fn=mock_evaluator,
        params=modules_config,
    )
    print(f"   Strategy loaded: {strategy}")

    # Create evolver with strategy's population_module
    evolver = Evolver(
        config=config,
        population_module=strategy.population_module,
        experiment=experiment,
        logger=logger,
        data_service=data_service,
        llm_client=mock_llm,
        env=env,
    )

    # Run evolution with strategy's rollout
    print("\n-> Running evolution (20 iterations)...")
    final_experiment = evolver.run(strategy.rollout)

    # Verify results
    assert final_experiment is not None, "Final experiment should not be None"
    assert final_experiment.current_iteration == 20, f"Should have completed 3 iterations, got {final_experiment.current_iteration}"
    assert len(final_experiment.archive) >= 20, f"Archive should have at least 4 programs (1 seed + 3 new), got {len(final_experiment.archive)}"

    # Get population
    population = final_experiment.get_all_programs()
    assert len(population) > 0, "Population should not be empty"

    # Find best program
    best_program = max(population, key=lambda p: p.combined_score or 0.0)

    print(f"\n[OK] Evolution completed successfully")
    print(f"    - Iterations completed: {final_experiment.current_iteration}")
    print(f"    - Archive size: {len(final_experiment.archive)}")
    print(f"    - Population size: {len(population)}")
    print(f"    - Best score: {best_program.combined_score:.4f}")
    print(f"    - Best program ID: {best_program.id}")

    # Print population scores
    print(f"\n    Population scores:")
    for prog in sorted(population, key=lambda p: p.combined_score or 0.0, reverse=True):
        print(f"      - {prog.id}: {prog.combined_score:.4f}")

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  FAMOU 2.0 SYSTEM TEST")
    print("=" * 70)
    print("\nRunning tests with mock LLM and mock evaluator...\n")

    # Set random seed for reproducibility
    random.seed(42)

    results = []

    try:
        # Test 1: RolloutEngine
        results.append(("RolloutEngine", test_rollout_engine()))
    except Exception as e:
        print(f"\n[FAILED] Test 1 (RolloutEngine): {e}")
        import traceback
        traceback.print_exc()
        results.append(("RolloutEngine", False))

    try:
        # Test 2: Evolver Loop
        results.append(("Evolver Loop", test_evolver_loop()))
    except Exception as e:
        print(f"\n[FAILED] Test 2 (Evolver Loop): {e}")
        import traceback
        traceback.print_exc()
        results.append(("Evolver Loop", False))

    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n  ALL TESTS PASSED!")
    else:
        print("\n  SOME TESTS FAILED!")
        exit(1)


if __name__ == "__main__":
    main()
