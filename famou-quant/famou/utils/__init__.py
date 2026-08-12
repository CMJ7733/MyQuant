"""
Utility functions for Famou 2.0.

ID Generation:
- generate_program_id: Generate unique program IDs
- generate_experiment_id: Generate unique experiment IDs
- generate_rollout_id: Generate unique rollout IDs
- get_timestamp: Get current timestamp

Code Parsing:
- extract_code_from_markdown: Extract code blocks from markdown
- evolve_code: Merge EVOLVE-BLOCK from child into parent code
- validate_python_syntax: Validate Python code syntax

Module Loading:
- load_function_from_file: Load a function from a Python file
- validate_evaluator_signature: Validate evaluator function signature

Package Management:
- install_packages: Install Python packages to target directory
- get_pythonpath_with_target: Get PYTHONPATH with target directory prepended
- clear_package_cache: Clear global package installation cache

Serialization:
- model_to_dict, model_to_json: Pydantic model to dict/JSON
- dict_to_model, json_to_model: Dict/JSON to Pydantic model
- load_model_from_file, save_model_to_file: File I/O for models

Math:
- cosine_distance, cosine_similarity: Vector distance/similarity
- normalize_vector: Unit vector normalization
- compute_mean_vector, compute_sum_vector: Vector aggregation
- stable_softmax: Numerically stable softmax with temperature
- euclidean_distance: Euclidean distance between vectors
- vectors_equal: Vector equality check
"""

from famou.utils.id_gen import (
    generate_experiment_id,
    generate_island_initial_program_id,
    generate_program_id,
    generate_rollout_id,
    get_timestamp,
    set_global_seed,
    timestamp_to_str,
)
from famou.utils.code_parser import (
    evolve_code,
    extract_and_validate_code,
    extract_code_from_markdown,
    extract_required_packages,
    validate_python_syntax,
)
from famou.utils.module_loader import (
    load_function_from_file,
    validate_evaluator_signature,
)
from famou.utils.package_manager import (
    clear_package_cache,
    install_packages,
)
from famou.utils.serialization import (
    dict_to_model,
    json_to_model,
    load_model_from_file,
    model_to_dict,
    model_to_json,
    save_model_to_file,
)
from famou.utils.math import (
    cosine_distance,
    cosine_similarity,
    normalize_vector,
    compute_mean_vector,
    compute_sum_vector,
    stable_softmax,
    euclidean_distance,
    vectors_equal,
)

__all__ = [
    # ID Generation
    "generate_program_id",
    "generate_experiment_id",
    "generate_island_initial_program_id",
    "generate_rollout_id",
    "get_timestamp",
    "set_global_seed",
    "timestamp_to_str",
    # Code Parsing
    "extract_code_from_markdown",
    "evolve_code",
    "extract_and_validate_code",
    "extract_required_packages",
    "validate_python_syntax",
    # Module Loading
    "load_function_from_file",
    "validate_evaluator_signature",
    # Package Management
    "install_packages",
    "get_pythonpath_with_target",
    "clear_package_cache",
    # Serialization
    "model_to_dict",
    "model_to_json",
    "dict_to_model",
    "json_to_model",
    "load_model_from_file",
    "save_model_to_file",
    # Math
    "cosine_distance",
    "cosine_similarity",
    "normalize_vector",
    "compute_mean_vector",
    "compute_sum_vector",
    "stable_softmax",
    "euclidean_distance",
    "vectors_equal",
]
