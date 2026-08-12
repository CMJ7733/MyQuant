"""
Module loader utility for dynamically loading functions from Python files.

This module provides a canonical way to load functions from arbitrary Python
files using spec_from_file_location, with proper error handling and validation.
"""

import importlib.util
import os
from pathlib import Path
from typing import Any, Callable, Tuple


def load_function_from_file(
    file_path: str,
    function_name: str,
    module_name: str = "_dynamic_module"
) -> Tuple[Callable, str]:
    """
    Load a function from a Python file using spec_from_file_location.

    This function:
    - Resolves relative paths to absolute paths
    - Validates that the file exists
    - Loads the module and executes it
    - Validates that the function exists and is callable
    - Returns both the function and the resolved absolute path

    Args:
        file_path: Path to the Python file (can be relative or absolute)
        function_name: Name of the function to extract from the module
        module_name: Name to use for the module in sys.modules (default: "_dynamic_module")

    Returns:
        Tuple of (function_object, resolved_absolute_path)

    Raises:
        FileNotFoundError: File doesn't exist at the given path
        ImportError: Module cannot be loaded or has import errors
        AttributeError: Function not found in module
        TypeError: Found object is not callable
    """
    # Resolve to absolute path early to avoid working directory issues
    try:
        abs_path = str(Path(file_path).resolve())
    except Exception as e:
        raise ValueError(f"Invalid file path '{file_path}': {e}") from e

    # Validate file exists
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Module file not found: {abs_path}\n"
            f"Original path: {file_path}"
        )

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Path exists but is not a file: {abs_path}"
        )

    # Create module spec
    try:
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
    except Exception as e:
        raise ImportError(f"Failed to create module spec for {abs_path}: {e}") from e

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load module from: {abs_path}\n"
            f"File may not be a valid Python module"
        )

    # Create module from spec
    try:
        mod = importlib.util.module_from_spec(spec)
    except Exception as e:
        raise ImportError(f"Failed to create module from spec: {e}") from e

    # Execute module (run all top-level code)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise ImportError(
            f"Failed to execute module {abs_path}: {e}\n"
            f"The module may have import errors or syntax errors"
        ) from e

    # Validate function exists
    if not hasattr(mod, function_name):
        # List available functions to help user debug
        available = [
            name for name in dir(mod)
            if not name.startswith('_') and callable(getattr(mod, name, None))
        ]
        available_str = ', '.join(available) if available else '(none)'

        raise AttributeError(
            f"Function '{function_name}' not found in {abs_path}.\n"
            f"Available callable functions: {available_str}"
        )

    fn = getattr(mod, function_name)

    # Validate it's callable
    if not callable(fn):
        raise TypeError(
            f"'{function_name}' in {abs_path} is not callable (got {type(fn).__name__})"
        )

    return fn, abs_path


def validate_evaluator_signature(
    fn: Callable,
    expected_params: list = None
) -> bool:
    """
    Validate that an evaluator function has the expected signature.

    Args:
        fn: Function to validate
        expected_params: Optional list of required parameter names

    Returns:
        True if signature is valid

    Raises:
        TypeError: Signature doesn't match expectations

    Example:
        >>> validate_evaluator_signature(evaluate_fn, ["program_path"])
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot inspect signature of {fn}: {e}") from e

    params = list(sig.parameters.keys())

    # Basic validation: should accept at least one positional argument
    if not params:
        raise TypeError(
            f"Evaluator function '{fn.__name__}' must accept at least one argument "
            f"(program_path)"
        )

    # If specific params expected, validate them
    if expected_params:
        for param in expected_params:
            if param not in params:
                raise TypeError(
                    f"Evaluator function '{fn.__name__}' is missing required "
                    f"parameter '{param}'. Found parameters: {params}"
                )

    return True
