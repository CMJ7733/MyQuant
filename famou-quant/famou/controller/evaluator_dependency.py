"""
Evaluator dependency resolution and deferred loading utilities.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from famou.infrastructure.llm import LLMClient
from famou.infrastructure.llm.base import (
    get_llm_max_retries,
    get_llm_max_tokens,
    get_llm_timeout,
)


PACKAGE_ALIASES: Dict[str, str] = {
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
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class DeferredEvaluator:
    """
    Deferred evaluator callable that imports the evaluator file only at call time.

    This avoids importing user evaluator code in the main process before
    evaluator dependencies have been prepared.
    """

    def __init__(
        self,
        evaluator_path: str,
        *,
        function_name: str = "evaluate",
        required_packages: Optional[List[str]] = None,
    ) -> None:
        self.evaluator_path = str(Path(evaluator_path).resolve())
        self.function_name = function_name
        self.required_packages = list(required_packages or [])
        self.__name__ = function_name

    def __call__(self, program_path: str, **kwargs: Any) -> Dict[str, Any]:
        spec = importlib.util.spec_from_file_location("famou_user_evaluator", self.evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load evaluator module from {self.evaluator_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fn = getattr(module, self.function_name, None)
        if fn is None:
            raise AttributeError(
                f"Evaluator file {self.evaluator_path} does not contain '{self.function_name}'"
            )
        if not callable(fn):
            raise TypeError(
                f"Object '{self.function_name}' in {self.evaluator_path} is not callable"
            )
        return fn(program_path, **kwargs)


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text()


def _normalize_package_name(value: str) -> Optional[str]:
    token = value.strip().strip("`'\"")
    if not token:
        return None
    token = token.split("#", 1)[0].strip()
    if not token:
        return None
    token = _PKG_SPLIT_RE.split(token, maxsplit=1)[0].strip()
    if not token:
        return None
    token = PACKAGE_ALIASES.get(token, token)
    token = token.replace("_", "-").strip().lower()
    return token or None


def _is_standard_library(module_name: str) -> bool:
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    return module_name in stdlib_names or module_name in {"__future__"}


def _is_local_module(module_name: str, evaluator_dir: Path) -> bool:
    parts = module_name.split(".")
    candidate = evaluator_dir
    for part in parts[:-1]:
        candidate = candidate / part
    leaf = parts[-1]
    return (candidate / f"{leaf}.py").exists() or (candidate / leaf / "__init__.py").exists()


def extract_static_evaluator_packages(evaluator_path: str) -> Dict[str, Any]:
    path = Path(evaluator_path).resolve()
    source = _read_text(path)
    tree = ast.parse(source, filename=str(path))

    raw_modules: List[str] = []
    skipped: List[Dict[str, str]] = []
    packages: List[str] = []

    for node in ast.walk(tree):
        module_name: Optional[str] = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".", 1)[0]
                raw_modules.append(alias.name)
                if _is_standard_library(module_name):
                    skipped.append({"module": alias.name, "reason": "stdlib"})
                    continue
                if _is_local_module(module_name, path.parent):
                    skipped.append({"module": alias.name, "reason": "local"})
                    continue
                normalized = _normalize_package_name(module_name)
                if normalized:
                    packages.append(normalized)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                skipped.append({"module": "." * node.level + (node.module or ""), "reason": "relative"})
                continue
            if not node.module:
                continue
            module_name = node.module.split(".", 1)[0]
            raw_modules.append(node.module)
            if _is_standard_library(module_name):
                skipped.append({"module": node.module, "reason": "stdlib"})
                continue
            if _is_local_module(module_name, path.parent):
                skipped.append({"module": node.module, "reason": "local"})
                continue
            normalized = _normalize_package_name(module_name)
            if normalized:
                packages.append(normalized)

    unique_packages = list(dict.fromkeys(packages))
    return {
        "packages": unique_packages,
        "raw_imports": raw_modules,
        "skipped": skipped,
    }


def _extract_json_blob(text: str) -> Dict[str, Any]:
    text = text.strip()
    match = _JSON_BLOCK_RE.search(text)
    candidate = match.group(1) if match else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(candidate[start : end + 1])


def resolve_llm_evaluator_packages(
    evaluator_path: str,
    evaluator_source: str,
    static_packages: List[str],
    llm_client: LLMClient,
) -> Dict[str, Any]:
    if llm_client is None:
        raise ValueError("LLM client is required for evaluator dependency resolution")

    system_prompt = (
        "You extract third-party Python package names used by an evaluator file. "
        "Return strict JSON only with keys: packages, reasoning. "
        "Rules: include only pip-installable third-party packages; exclude Python stdlib, "
        "exclude local project modules, exclude versions, extras, and constraints. "
        "Output package names only."
    )
    user_prompt = json.dumps(
        {
            "task": "Extract third-party packages required by this evaluator file.",
            "evaluator_path": str(Path(evaluator_path).resolve()),
            "static_candidates": static_packages,
            "evaluator_source": evaluator_source,
        },
        ensure_ascii=False,
        indent=2,
    )
    attempts: List[Dict[str, Any]] = []
    max_attempts = get_llm_max_retries(llm_client)
    max_tokens = get_llm_max_tokens(llm_client)
    timeout = get_llm_timeout(llm_client)

    for attempt_index in range(1, max_attempts + 1):
        attempt_prompt = user_prompt
        if attempt_index > 1:
            attempt_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Your previous response could not be parsed. "
                "Return exactly one JSON object and nothing else."
            )

        started_at = time.time()
        response = None
        completed_at = started_at
        request_payload = {
            "system": system_prompt,
            "prompt": attempt_prompt,
            "attempt": attempt_index,
            "max_attempts": max_attempts,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        try:
            response = llm_client.generate(
                prompt=attempt_prompt,
                system=system_prompt,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            completed_at = time.time()

            parsed = _extract_json_blob(response.text)
            raw_packages = parsed.get("packages") or parsed.get("required_packages") or []
            if not isinstance(raw_packages, list):
                raise ValueError("LLM dependency response must contain a list under 'packages'")

            packages: List[str] = []
            for item in raw_packages:
                if not isinstance(item, str):
                    continue
                normalized = _normalize_package_name(item)
                if normalized:
                    packages.append(normalized)

            response_payload = {
                "text": response.text,
                "raw_response": response.raw_response,
                "usage": response.usage,
                "model": response.model,
                "provider": response.provider,
                "finish_reason": response.finish_reason,
                "started_at": response.started_at or started_at,
                "completed_at": response.completed_at or completed_at,
                "latency_ms": response.latency_ms or int((completed_at - started_at) * 1000),
            }
            attempts.append(
                {
                    "attempt": attempt_index,
                    "success": True,
                    "request": request_payload,
                    "response": response_payload,
                    "parsed": {
                        "packages": list(dict.fromkeys(packages)),
                        "reasoning": str(parsed.get("reasoning", "")).strip(),
                    },
                }
            )

            return {
                "success": True,
                "packages": list(dict.fromkeys(packages)),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
                "request": request_payload,
                "response": response_payload,
                "attempts": attempts,
                "attempt_count": attempt_index,
            }
        except Exception as exc:
            completed_at = time.time()
            response_payload = None
            if response is not None:
                response_payload = {
                    "text": response.text,
                    "raw_response": response.raw_response,
                    "usage": response.usage,
                    "model": response.model,
                    "provider": response.provider,
                    "finish_reason": response.finish_reason,
                    "started_at": response.started_at or started_at,
                    "completed_at": response.completed_at or completed_at,
                    "latency_ms": response.latency_ms or int((completed_at - started_at) * 1000),
                }

            attempts.append(
                {
                    "attempt": attempt_index,
                    "success": False,
                    "request": request_payload,
                    "response": response_payload,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )

    return {
        "success": False,
        "packages": [],
        "reasoning": "",
        "request": attempts[-1]["request"] if attempts else None,
        "response": attempts[-1].get("response") if attempts else None,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "fallback_reason": (
            attempts[-1]["error"]["message"]
            if attempts and attempts[-1].get("error")
            else "llm_dependency_resolution_failed"
        ),
    }


def extract_static_init_packages(
    source: str,
    source_path: Optional[Path] = None,
) -> List[str]:
    """Static AST analysis for an init program given its source string.

    Args:
        source: Source code of the init program.
        source_path: Optional file path used for local-module detection.

    Returns:
        Deduplicated list of normalized third-party package names.
    """
    try:
        tree = ast.parse(source, filename=str(source_path or "<init>"))
    except SyntaxError:
        return []

    packages: List[str] = []
    parent_dir = source_path.parent if source_path else None

    for node in ast.walk(tree):
        module_name: Optional[str] = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".", 1)[0]
                if _is_standard_library(module_name):
                    continue
                if parent_dir and _is_local_module(module_name, parent_dir):
                    continue
                normalized = _normalize_package_name(module_name)
                if normalized:
                    packages.append(normalized)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if not node.module:
                continue
            module_name = node.module.split(".", 1)[0]
            if _is_standard_library(module_name):
                continue
            if parent_dir and _is_local_module(module_name, parent_dir):
                continue
            normalized = _normalize_package_name(module_name)
            if normalized:
                packages.append(normalized)

    return list(dict.fromkeys(packages))


def resolve_llm_init_packages_batch(
    files: List[Dict[str, str]],
    static_packages: List[str],
    llm_client: LLMClient,
) -> Dict[str, Any]:
    """Multi-file LLM package extraction for a batch of init programs.

    Args:
        files: List of dicts with keys ``path`` (str) and ``source`` (str).
        static_packages: Packages already found via static analysis (hint for LLM).
        llm_client: LLM client to use.

    Returns:
        Same schema as ``resolve_llm_evaluator_packages``.
    """
    system_prompt = (
        "You extract third-party Python package names used by one or more program files. "
        "Return strict JSON only with keys: packages, reasoning. "
        "Rules: include only pip-installable third-party packages; exclude Python stdlib, "
        "exclude local project modules, exclude versions, extras, and constraints. "
        "Output package names only, combined across all provided files."
    )
    user_prompt = json.dumps(
        {
            "task": "Extract all third-party packages required by these program files (combined).",
            "static_candidates": static_packages,
            "files": [
                {"path": f["path"], "source": f["source"]} for f in files
            ],
        },
        ensure_ascii=False,
        indent=2,
    )

    attempts: List[Dict[str, Any]] = []
    max_attempts = get_llm_max_retries(llm_client)
    max_tokens = get_llm_max_tokens(llm_client)
    timeout = get_llm_timeout(llm_client)

    for attempt_index in range(1, max_attempts + 1):
        attempt_prompt = user_prompt
        if attempt_index > 1:
            attempt_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Your previous response could not be parsed. "
                "Return exactly one JSON object and nothing else."
            )

        started_at = time.time()
        response = None
        completed_at = started_at
        request_payload = {
            "system": system_prompt,
            "prompt": attempt_prompt,
            "attempt": attempt_index,
            "max_attempts": max_attempts,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "file_count": len(files),
        }

        try:
            response = llm_client.generate(
                prompt=attempt_prompt,
                system=system_prompt,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            completed_at = time.time()

            parsed = _extract_json_blob(response.text)
            raw_packages = parsed.get("packages") or parsed.get("required_packages") or []
            if not isinstance(raw_packages, list):
                raise ValueError("LLM response must contain a list under 'packages'")

            packages: List[str] = []
            for item in raw_packages:
                if not isinstance(item, str):
                    continue
                normalized = _normalize_package_name(item)
                if normalized:
                    packages.append(normalized)

            response_payload = {
                "text": response.text,
                "raw_response": response.raw_response,
                "usage": response.usage,
                "model": response.model,
                "provider": response.provider,
                "finish_reason": response.finish_reason,
                "started_at": response.started_at or started_at,
                "completed_at": response.completed_at or completed_at,
                "latency_ms": response.latency_ms or int((completed_at - started_at) * 1000),
            }
            attempts.append({
                "attempt": attempt_index,
                "success": True,
                "request": request_payload,
                "response": response_payload,
                "parsed": {
                    "packages": list(dict.fromkeys(packages)),
                    "reasoning": str(parsed.get("reasoning", "")).strip(),
                },
            })
            return {
                "success": True,
                "packages": list(dict.fromkeys(packages)),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
                "request": request_payload,
                "response": response_payload,
                "attempts": attempts,
                "attempt_count": attempt_index,
            }

        except Exception as exc:
            completed_at = time.time()
            response_payload = None
            if response is not None:
                response_payload = {
                    "text": response.text,
                    "raw_response": response.raw_response,
                    "usage": response.usage,
                    "model": response.model,
                    "provider": response.provider,
                    "finish_reason": response.finish_reason,
                    "started_at": response.started_at or started_at,
                    "completed_at": response.completed_at or completed_at,
                    "latency_ms": response.latency_ms or int((completed_at - started_at) * 1000),
                }
            attempts.append({
                "attempt": attempt_index,
                "success": False,
                "request": request_payload,
                "response": response_payload,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })

    return {
        "success": False,
        "packages": [],
        "reasoning": "",
        "request": attempts[-1]["request"] if attempts else None,
        "response": attempts[-1].get("response") if attempts else None,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "fallback_reason": (
            attempts[-1]["error"]["message"]
            if attempts and attempts[-1].get("error")
            else "llm_init_batch_resolution_failed"
        ),
    }


def _make_batches(items: List[Any], max_batch_size: int = 5) -> List[List[Any]]:
    """Split items into evenly distributed batches of at most max_batch_size."""
    n = len(items)
    if n == 0:
        return []
    num_batches = math.ceil(n / max_batch_size)
    base, remainder = divmod(n, num_batches)
    batches: List[List[Any]] = []
    start = 0
    for i in range(num_batches):
        size = base + (1 if i < remainder else 0)
        batches.append(items[start: start + size])
        start += size
    return batches


def resolve_init_program_dependencies(
    programs: List[Any],
    program_paths: List[str],
    llm_client: LLMClient,
    max_batch_size: int = 5,
) -> Dict[str, Any]:
    """Resolve packages for init programs via static analysis + batched LLM.

    Static analysis is done per file; LLM calls are batched (max ``max_batch_size``
    files per call, evenly distributed). Results are merged as a union and written
    back to each ``program.required_packages``.

    Args:
        programs: List of Program objects (must have ``.code``, ``.id`` attributes).
        program_paths: File paths corresponding to each program (same order).
        llm_client: LLM client for package extraction.
        max_batch_size: Maximum number of files per LLM batch call (default 5).

    Returns:
        Dict with keys:
            ``final_packages`` – deduplicated union of all extracted packages.
            ``static``         – per-file static analysis results.
            ``llm_batches``    – list of per-batch LLM results.
    """
    # ── Static analysis (per file) ────────────────────────────────────────────
    static_results: List[Dict[str, Any]] = []
    all_static_packages: List[str] = []

    for program, path_str in zip(programs, program_paths):
        path = Path(path_str) if path_str else None
        pkgs = extract_static_init_packages(program.code, source_path=path)
        static_results.append({"program_id": program.id, "packages": pkgs})
        all_static_packages.extend(pkgs)

    static_union = list(dict.fromkeys(all_static_packages))

    # Set static packages on each program immediately as a baseline
    for program, sr in zip(programs, static_results):
        program.required_packages = list(sr["packages"])

    # ── Batched LLM extraction ────────────────────────────────────────────────
    file_infos: List[Dict[str, str]] = [
        {"path": str(program_paths[i]), "source": programs[i].code}
        for i in range(len(programs))
    ]
    batches = _make_batches(file_infos, max_batch_size)

    llm_batch_results: List[Dict[str, Any]] = []
    all_llm_packages: List[str] = []

    for batch_index, batch in enumerate(batches):
        result = resolve_llm_init_packages_batch(
            files=batch,
            static_packages=static_union,
            llm_client=llm_client,
        )
        result["batch_index"] = batch_index
        result["batch_file_count"] = len(batch)
        llm_batch_results.append(result)
        all_llm_packages.extend(result.get("packages") or [])

    llm_union = list(dict.fromkeys(all_llm_packages))
    final_packages = list(dict.fromkeys(static_union + llm_union))

    return {
        "final_packages": final_packages,
        "static": {
            "per_file": static_results,
            "union": static_union,
        },
        "llm_batches": llm_batch_results,
        "llm_union": llm_union,
        "batch_count": len(batches),
        "llm_success": all(r.get("success") for r in llm_batch_results),
    }


def resolve_evaluator_dependencies(
    evaluator_path: str,
    llm_client: LLMClient,
) -> Dict[str, Any]:
    path = Path(evaluator_path).resolve()
    source = _read_text(path)
    static_result = extract_static_evaluator_packages(str(path))
    llm_result = resolve_llm_evaluator_packages(
        evaluator_path=str(path),
        evaluator_source=source,
        static_packages=static_result["packages"],
        llm_client=llm_client,
    )

    llm_success = bool(llm_result.get("success"))
    if llm_success:
        final_packages = list(
            dict.fromkeys(static_result["packages"] + llm_result["packages"])
        )
        merge_strategy = "static_union_llm"
    else:
        final_packages = list(dict.fromkeys(static_result["packages"]))
        merge_strategy = "fallback_to_static"

    return {
        "meta": {
            "evaluator_path": str(path),
            "evaluator_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "resolved_at": time.time(),
        },
        "static": static_result,
        "llm": llm_result,
        "decision": {
            "final_packages": final_packages,
            "merge_strategy": merge_strategy,
            "llm_success": llm_success,
            "llm_attempt_count": llm_result.get("attempt_count", 0),
            "fallback_to_static": not llm_success,
            "fallback_reason": llm_result.get("fallback_reason"),
        },
    }


def install_packages_into_current_env(packages: Iterable[str]) -> Dict[str, Any]:
    normalized = [pkg for pkg in dict.fromkeys(packages) if pkg]
    if not normalized:
        return {"mode": "current_env", "packages": [], "success": True}

    cmd = [sys.executable, "-m", "pip", "install", *normalized]
    started_at = time.time()
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "mode": "current_env",
            "packages": normalized,
            "success": True,
            "command": cmd,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "started_at": started_at,
            "completed_at": time.time(),
        }
    except subprocess.CalledProcessError as exc:
        return {
            "mode": "current_env",
            "packages": normalized,
            "success": False,
            "command": cmd,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
            "started_at": started_at,
            "completed_at": time.time(),
            "returncode": exc.returncode,
        }


def prepare_evaluator_environment(
    *,
    evaluator_path: str,
    llm_client: LLMClient,
    env: Any,
    storage: Any,
    experiment_id: str,
    logger: Optional[Any] = None,
    prefer_env_install: bool = True,
) -> Dict[str, Any]:
    trace = resolve_evaluator_dependencies(evaluator_path, llm_client)
    packages = list(trace["decision"]["final_packages"])

    if trace["decision"].get("fallback_to_static") and logger is not None:
        logger.warning(
            "LLM evaluator dependency extraction failed; falling back to static packages",
            llm_attempt_count=trace["decision"].get("llm_attempt_count"),
            fallback_reason=trace["decision"].get("fallback_reason"),
            static_packages=packages,
        )

    install_trace: Dict[str, Any]
    env_is_factory = bool(
        env is not None
        and hasattr(env, "create")
        and not hasattr(env, "execute_with_evaluator")
    )

    if packages and env_is_factory:
        install_trace = {
            "mode": "deferred_runtime",
            "packages": packages,
            "success": True,
            "prepared": False,
            "deferred": True,
        }
    elif packages and prefer_env_install and hasattr(env, "prepare_dependencies"):
        install_trace = env.prepare_dependencies(packages)
    elif packages and prefer_env_install and hasattr(env, "create"):
        prepared_env = env.create()
        try:
            if hasattr(prepared_env, "prepare_dependencies"):
                install_trace = prepared_env.prepare_dependencies(packages)
                install_trace["factory_prepared"] = True
            else:
                install_trace = install_packages_into_current_env(packages)
        finally:
            if hasattr(prepared_env, "close"):
                prepared_env.close()
    else:
        install_trace = install_packages_into_current_env(packages)

    trace["install"] = install_trace
    trace["decision"]["loader"] = "deferred_evaluator"
    trace["decision"]["install_success"] = bool(install_trace.get("success", False))
    trace["decision"]["continue_after_install_failure"] = not bool(
        install_trace.get("success", False)
    )

    if storage is not None and hasattr(storage, "save_json_artifact"):
        storage.save_json_artifact(
            experiment_id,
            "evaluator_dependency_resolution.json",
            trace,
        )
        if logger:
            logger.info(
                "Persisted evaluator dependency trace",
                filename="evaluator_dependency_resolution.json",
            )

    if install_trace.get("deferred") and logger is not None:
        logger.info(
            "Evaluator dependencies will be prepared lazily in the execution environment",
            packages=install_trace.get("packages"),
            install_mode=install_trace.get("mode"),
        )
    elif not install_trace.get("success", False):
        if logger is not None:
            logger.warning(
                "Failed to prepare evaluator dependencies; continuing without eager install",
                install_mode=install_trace.get("mode"),
                packages=install_trace.get("packages"),
                error=install_trace.get("stderr")
                or install_trace.get("error")
                or "unknown error",
            )

    return trace
