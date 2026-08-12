"""
Code extraction and validation utilities for Famou 2.0.

Provides functions to extract code from LLM responses (which often include
markdown code blocks) and validate Python syntax.
"""

import json
import re
import sys
from typing import Dict, List, Optional, Tuple


PACKAGE_ALIASES = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
    "OpenGL": "pyopengl",
    "dateutil": "python-dateutil",
}

_PKG_SPLIT_RE = re.compile(r"[<>=!~\[\]\s,;]+")



def extract_code_from_markdown(text: str, language: str = "python") -> str:
    """
    Extract code from markdown code blocks.

    Handles various markdown formats:
    - ```python\\n...\\n```
    - ```py\\n...\\n```
    - ```\\n...\\n```

    If no code blocks are found, returns the entire text (assuming the LLM
    returned raw code without markdown).

    Args:
        text: The text containing code (potentially in markdown blocks)
        language: The programming language to look for (default: "python")

    Returns:
        Extracted code as string

    Example:
        >>> text = "```python\\ndef foo():\\n    pass\\n```"
        >>> extract_code_from_markdown(text)
        'def foo():\\n    pass'
    """
    # Try language-specific patterns first
    patterns = [
        rf"```{language}\n(.*?)```",  # ```python
        r"```py\n(.*?)```",  # ```py
        r"```\n(.*?)```",  # ```
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return the first match (usually there's only one code block)
            return matches[0].strip()

    # Fallback: return entire text if no code blocks found
    # This handles cases where the LLM returns raw code without markdown
    return text.strip()


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if code is syntactically valid Python.

    Args:
        code: Python code string to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if syntax is valid, False otherwise
        - error_message: Error description if invalid, None otherwise

    Example:
        >>> validate_python_syntax("def foo(): pass")
        (True, None)
        >>> validate_python_syntax("def foo(")
        (False, "SyntaxError: ...")
    """
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def extract_and_validate_code(
    text: str, language: str = "python"
) -> Tuple[str, bool, Optional[str]]:
    """
    Extract code from text and validate it in one step.

    Args:
        text: Text containing code (potentially in markdown)
        language: Programming language (default: "python")

    Returns:
        Tuple of (code, is_valid, error_message)

    Example:
        >>> text = "```python\\ndef foo():\\n    pass\\n```"
        >>> code, valid, err = extract_and_validate_code(text)
        >>> code
        'def foo():\\n    pass'
        >>> valid
        True
    """
    code = extract_code_from_markdown(text, language)

    if language == "python":
        is_valid, error = validate_python_syntax(code)
        return code, is_valid, error

    # For other languages, skip validation
    return code, True, None


def extract_function_name(code: str, default: str = "main") -> str:
    """
    Extract the first function name from Python code.

    Args:
        code: Python code string
        default: Default function name if none found

    Returns:
        Function name

    Example:
        >>> extract_function_name("def sort(arr):\\n    pass")
        'sort'
        >>> extract_function_name("x = 1", default="main")
        'main'
    """
    # Pattern to match: def function_name(
    pattern = r"def\s+(\w+)\s*\("
    matches = re.findall(pattern, code)

    if matches:
        return matches[0]

    return default


def count_lines_of_code(code: str, ignore_empty: bool = True) -> int:
    """
    Count lines of code.

    Args:
        code: Python code string
        ignore_empty: Whether to ignore empty lines and comments

    Returns:
        Number of lines

    Example:
        >>> count_lines_of_code("def foo():\\n    pass\\n")
        2
    """
    lines = code.split("\n")

    if not ignore_empty:
        return len(lines)

    # Count only non-empty, non-comment lines
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1

    return count


def has_single_evolve_block(code: str, language: str = "python") -> bool:
    """Return True when code contains exactly one complete EVOLVE-BLOCK pair."""
    lang = language.lower().strip()
    comment = "#" if lang in ("python", "py") else "//"
    pattern = rf"{re.escape(comment)}\s*EVOLVE-BLOCK-START[\s\S]*?{re.escape(comment)}\s*EVOLVE-BLOCK-END"
    return len(re.findall(pattern, code)) == 1


def evolve_code(child_code: str, parent_code: str, language: str = "python") -> str:
    """
    Evolve code inside EVOLVE BLOCK, while keeping other parts unchanged.

    Args:
        child_code: Child code string containing new EVOLVE-BLOCK
        parent_code: Parent code string
        language: Programming language (determines comment prefix)

    Returns:
        New evolved code string
    """
    lang = language.lower().strip()
    if lang in ("python", "py"):
        comment = "#"
    else:
        comment = "//"

    pattern = rf"({re.escape(comment)}\s*EVOLVE-BLOCK-START[\s\S]*?{re.escape(comment)}\s*EVOLVE-BLOCK-END)"
    
    # 1. 尝试从 child_code 中提取新块
    child_match = re.search(pattern, child_code)
    if not child_match:
        return child_code
        
    # 2. 检查 parent_code 是否包含旧块 (如果父代码没这个块，无法替换，按原逻辑返回 child)
    if not re.search(pattern, parent_code):
        return child_code
        
    # 3. 提取新块的内容
    new_block_content = child_match.group(1)
    
    # 4. 执行替换
    # 如果需要完全模仿原代码的 "rstrip() + \n\n" 格式化逻辑，可以在这里微调
    # 这里使用最通用的直接替换，保留父代码原本的缩进和周边结构
    merged_content = re.sub(pattern, lambda m: new_block_content, parent_code, count=1)
    
    return merged_content


def extract_required_packages(text: str) -> List[str]:
    """
    Extract required packages from markdown code block.

    Looks for ```required_packages block and parses package names.
    Handles version specifiers, extras, and comments.

    Args:
        text: The text containing the required_packages block

    Returns:
        List of package names (without version specifiers)

    Example:
        >>> text = '```required_packages\\nnumpy>=1.20\\nmatplotlib\\nscipy[stats]\\n```'
        >>> extract_required_packages(text)
        ['numpy', 'matplotlib', 'scipy']
    """
    pattern = r"```required_packages\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        packages_text = match.group(1).strip()
        return sanitize_required_packages(packages_text.splitlines()) or []
    return []


def sanitize_required_packages(packages: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize required packages and drop stdlib or malformed entries."""
    if not packages:
        return None

    cleaned: List[str] = []
    seen = set()
    stdlib_names = getattr(sys, "stdlib_module_names", set())

    for item in packages:
        if not isinstance(item, str):
            continue
        token = item.strip().strip("`'\"")
        if not token or token.startswith("#"):
            continue
        token = token.split("#", 1)[0].strip()
        if not token:
            continue

        raw_name = _PKG_SPLIT_RE.split(token, maxsplit=1)[0].strip()
        if not raw_name:
            continue

        module_name = raw_name.split(".", 1)[0].strip()
        module_key = module_name.replace("-", "_")
        if module_key in stdlib_names or module_key == "__future__":
            continue

        normalized = PACKAGE_ALIASES.get(module_name, module_name)
        normalized = normalized.replace("_", "-").strip().lower()
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(normalized)

    return cleaned or None


def extract_json(response_text: str) -> Dict:
    """
    Extracts a JSON object from a string. It handles cases where the JSON is
    embedded in a larger text or a markdown code block.

    Args:
        response_text (str): The raw text response from the LLM.

    Returns:
        dict or list or None: The parsed JSON object if found and valid, 
                              otherwise None.
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            # Try to parse the extracted string as JSON
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If it fails, we'll try the next method
            pass
    try:
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = response_text[start_index:end_index + 1]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        # If parsing fails, it's not a valid JSON object.
        raise e
    
    # Return None if no valid JSON could be extracted
    return {}
