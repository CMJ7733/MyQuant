"""Static AST audit: structural and safety gate before anything is executed.

Two jobs, both of which have to happen *before* the sandbox:

1. **Contract** — exactly one top-level function, sane signature, a docstring.
2. **Safety** — an import allow-list and a ban on the escape hatches
   (``eval``/``exec``/``__import__``, ``open``, dunder attribute walks).  The
   sandbox in :mod:`cogalpha.quality.sandbox` is the enforcement layer; this is
   the layer that keeps the obviously-hostile out of it, so a rejection costs no
   process spawn.

Neither job overlaps with :mod:`cogalpha.quality.leakage`: causality is a separate
concern with its own stage, because a look-ahead factor is *safe* to run — it is
just worthless.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

#: Callables that let generated code reach outside its own expression.
FORBIDDEN_NAMES: Set[str] = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "memoryview",
    "exit",
    "quit",
}

#: Modules that are never legitimate inside an alpha, even if a checker slipped.
FORBIDDEN_MODULES: Set[str] = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "importlib",
    "builtins",
    "ctypes",
    "pickle",
    "marshal",
    "requests",
    "urllib",
    "urllib3",
    "http",
    "ftplib",
    "smtplib",
    "multiprocessing",
    "threading",
    "asyncio",
    "tempfile",
    "glob",
    "io",
    "code",
    "codeop",
    "types",
    "gc",
    "inspect",
    "atexit",
    "signal",
    "resource",
}

#: Attribute names whose mere appearance signals a sandbox-escape attempt.
FORBIDDEN_ATTRS: Set[str] = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__builtins__",
    "__dict__",
    "__mro__",
    "__reduce__",
    "__reduce_ex__",
    "__getattribute__",
    "__loader__",
    "__spec__",
}

#: pandas/numpy methods that write to disk — banned so a factor cannot leave state.
FORBIDDEN_IO_METHODS: Set[str] = {
    "to_csv",
    "to_pickle",
    "to_parquet",
    "to_hdf",
    "to_feather",
    "to_sql",
    "to_json",
    "to_excel",
    "save",
    "savez",
    "savetxt",
    "tofile",
    "fromfile",
    "load",
    "system",
    "popen",
}


@dataclass
class AuditResult:
    """Outcome of the static audit."""

    ok: bool
    issues: List[str] = field(default_factory=list)
    function_name: Optional[str] = None
    has_docstring: bool = False
    imports: List[str] = field(default_factory=list)

    @property
    def detail(self) -> str:
        """Findings joined into one line, for the CheckReport detail field."""
        return "; ".join(self.issues) if self.issues else "clean"


def audit_code(
    code: str,
    allowed_imports: Sequence[str] = ("numpy", "np", "pandas", "pd", "math", "scipy", "talib"),
    require_docstring: bool = True,
    max_chars: int = 20_000,
) -> AuditResult:
    """Audit one alpha's source.

    ``allowed_imports`` is matched on the *root* module, so ``scipy.stats`` is
    permitted when ``scipy`` is.  Aliases in the list (``np``, ``pd``) are ignored
    for matching; they exist so a config can be written the way people think.
    """
    issues: List[str] = []

    if len(code) > max_chars:
        return AuditResult(
            ok=False,
            issues=[f"code is {len(code)} chars, over the {max_chars} limit"],
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return AuditResult(
            ok=False,
            issues=[f"SyntaxError at line {exc.lineno}: {exc.msg}"],
        )

    roots = {m.split(".")[0] for m in allowed_imports}

    # ------------------------------------------------------------- structure
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    others = [
        n
        for n in tree.body
        if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Expr))
    ]

    if len(funcs) == 0:
        issues.append("no top-level function defined")
    elif len(funcs) > 1:
        issues.append(
            f"expected exactly one top-level function, found {len(funcs)}: "
            + ", ".join(f.name for f in funcs)
        )

    if others:
        kinds = sorted({type(n).__name__ for n in others})
        issues.append(f"top-level statements other than the function: {kinds}")

    fn_name: Optional[str] = None
    has_doc = False
    if funcs:
        fn = funcs[0]
        fn_name = fn.name
        has_doc = ast.get_docstring(fn) is not None
        if require_docstring and not has_doc:
            issues.append("function has no docstring stating rationale and formula")
        n_pos = len(fn.args.args)
        n_required = n_pos - len(fn.args.defaults)
        if n_pos == 0:
            issues.append("function takes no arguments; it must accept the OHLCV frame")
        elif n_required > 1:
            issues.append(
                f"function requires {n_required} arguments; it must be callable as f(df)"
            )
        if fn.args.vararg or fn.args.kwonlyargs:
            issues.append("function must not use *args or keyword-only arguments")
        if not any(isinstance(n, ast.Return) for n in ast.walk(fn)):
            issues.append("function never returns a value")

    # ---------------------------------------------------------------- imports
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                imports.append(alias.name)
                if root in FORBIDDEN_MODULES:
                    issues.append(f"forbidden import '{alias.name}'")
                elif root not in roots:
                    issues.append(
                        f"import '{alias.name}' is outside the allow-list {sorted(roots)}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            imports.append(module)
            if node.level and node.level > 0:
                issues.append("relative imports are not allowed")
            elif root in FORBIDDEN_MODULES:
                issues.append(f"forbidden import from '{module}'")
            elif root and root not in roots:
                issues.append(
                    f"import from '{module}' is outside the allow-list {sorted(roots)}"
                )

    # ------------------------------------------------------------------ names
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            issues.append(f"use of forbidden builtin '{node.id}'")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                issues.append(f"access to forbidden attribute '{node.attr}'")
            elif node.attr in FORBIDDEN_IO_METHODS:
                issues.append(f"call to I/O method '{node.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            issues.append("global/nonlocal statements are not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await)):
            issues.append("async code is not allowed")
        elif isinstance(node, ast.Lambda):
            # Lambdas are fine and common in pandas apply; no issue. Kept explicit
            # so a future reader does not add a ban by accident.
            continue

    return AuditResult(
        ok=not issues,
        issues=issues,
        function_name=fn_name,
        has_docstring=has_doc,
        imports=sorted(set(imports)),
    )
