"""
Pydantic serialization utilities for Famou 2.0.

Provides helper functions for serializing and deserializing Pydantic models
to/from various formats (JSON, dict, etc.).
"""

import json
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, List

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def model_to_dict(model: BaseModel, exclude_none: bool = True) -> Dict[str, Any]:
    """
    Convert a Pydantic model to a dictionary.

    Args:
        model: Pydantic model instance
        exclude_none: Whether to exclude None values (default: True)

    Returns:
        Dictionary representation

    Example:
        >>> from pydantic import BaseModel
        >>> class Example(BaseModel):
        ...     name: str
        ...     value: int = None
        >>> model_to_dict(Example(name="test"))
        {'name': 'test'}
    """
    return model.model_dump(exclude_none=exclude_none)


def model_to_json(
    model: BaseModel, indent: int = 2, exclude_none: bool = True
) -> str:
    """
    Convert a Pydantic model to JSON string.

    Args:
        model: Pydantic model instance
        indent: JSON indentation (default: 2)
        exclude_none: Whether to exclude None values (default: True)

    Returns:
        JSON string

    Example:
        >>> from pydantic import BaseModel
        >>> class Example(BaseModel):
        ...     name: str
        >>> model_to_json(Example(name="test"))
        '{\\n  "name": "test"\\n}'
    """
    return model.model_dump_json(indent=indent, exclude_none=exclude_none)


def dict_to_model(data: Dict[str, Any], model_class: Type[T]) -> T:
    """
    Create a Pydantic model from a dictionary.

    Args:
        data: Dictionary with model data
        model_class: The Pydantic model class

    Returns:
        Model instance

    Example:
        >>> from pydantic import BaseModel
        >>> class Example(BaseModel):
        ...     name: str
        >>> dict_to_model({"name": "test"}, Example)
        Example(name='test')
    """
    return model_class.model_validate(data)


def json_to_model(json_str: str, model_class: Type[T]) -> T:
    """
    Create a Pydantic model from a JSON string.

    Args:
        json_str: JSON string
        model_class: The Pydantic model class

    Returns:
        Model instance

    Example:
        >>> from pydantic import BaseModel
        >>> class Example(BaseModel):
        ...     name: str
        >>> json_to_model('{"name": "test"}', Example)
        Example(name='test')
    """
    return model_class.model_validate_json(json_str)


def load_model_from_file(file_path: Path, model_class: Type[T]) -> T:
    """
    Load a Pydantic model from a JSON file.

    Args:
        file_path: Path to JSON file
        model_class: The Pydantic model class

    Returns:
        Model instance

    Example:
        >>> from pathlib import Path
        >>> from pydantic import BaseModel
        >>> class Example(BaseModel):
        ...     name: str
        >>> # Assuming 'example.json' contains {"name": "test"}
        >>> load_model_from_file(Path("example.json"), Example)
        Example(name='test')
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_str = f.read()
    return json_to_model(json_str, model_class)


def save_model_to_file(model: BaseModel, file_path: Path, indent: int = 2) -> None:
    """
    Save a Pydantic model to a JSON file.

    Args:
        model: Pydantic model instance
        file_path: Path to save JSON file
        indent: JSON indentation (default: 2)

    Example:
        >>> from pathlib import Path
        >>> from pydantic import BaseModel
        >>> class Example(BaseModel):
        ...     name: str
        >>> save_model_to_file(Example(name="test"), Path("example.json"))
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(model_to_json(model, indent=indent))


def models_to_jsonl(models: List[BaseModel]) -> str:
    """
    Convert a list of Pydantic models to JSONL format (one JSON per line).

    Args:
        models: List of Pydantic model instances

    Returns:
        JSONL string

    Example:
        >>> from pydantic import BaseModel
        >>> class Example(BaseModel):
        ...     name: str
        >>> models = [Example(name="a"), Example(name="b")]
        >>> print(models_to_jsonl(models))
        {"name": "a"}
        {"name": "b"}
    """
    lines = []
    for model in models:
        # Use compact JSON (no indentation) for JSONL
        json_line = model.model_dump_json(exclude_none=True)
        lines.append(json_line)
    return "\n".join(lines)


def jsonl_to_models(jsonl_str: str, model_class: Type[T]) -> List[T]:
    """
    Parse JSONL string to a list of Pydantic models.

    Args:
        jsonl_str: JSONL string (one JSON per line)
        model_class: The Pydantic model class

    Returns:
        List of model instances

    Example:
        >>> from pydantic import BaseModel
        >>> class Example(BaseModel):
        ...     name: str
        >>> jsonl = '{"name": "a"}\\n{"name": "b"}'
        >>> jsonl_to_models(jsonl, Example)
        [Example(name='a'), Example(name='b')]
    """
    models = []
    for line in jsonl_str.strip().split("\n"):
        if line.strip():  # Skip empty lines
            models.append(json_to_model(line, model_class))
    return models
