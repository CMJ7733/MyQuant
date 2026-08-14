"""Static AST audit: structural and safety gate before anything is executed.

Three jobs, all of which have to happen *before* the sandbox:

1. **Contract** — exactly one top-level function, sane signature, a docstring.
2. **Safety** — an import allow-list and a ban on the escape hatches
   (``eval``/``exec``/``__import__``, ``open``, dunder attribute walks).  The
   sandbox in :mod:`cogalpha.quality.sandbox` is the enforcement layer; this is
   the layer that keeps the obviously-hostile out of it, so a rejection costs no
   process spawn.
3. **Honest missing data** — a ban on constant fills.  This one is here rather
   than in :mod:`cogalpha.quality.numeric` because it is the *only* stage that can
   see it: ``fillna(0)`` is undetectable downstream by construction, since its
   whole effect is to make the missing values stop looking missing.

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


#: Calls that replace missing values with a constant.
#:
#: Rejected because the numerical-stability gate *measures* missingness:
#: ``nan_ratio_limit`` is the check that says "this alpha could not compute a value
#: often enough to be usable", and ``fillna(0)`` drives the measured ratio to zero
#: without supplying a single value.  The second harm is cross-sectional: a stock
#: with no history and a stock genuinely reading zero end up on the same value, so
#: the ranking cannot separate them and they join one large tie group (see
#: ``NumericReport.mean_tie_ratio``).
#:
#: Method-based fills (``ffill``, ``bfill``, ``fillna(method=...)``) are *not*
#: listed: they propagate an observed value rather than inventing one, which is a
#: defensible modelling choice with its own causality story.
CONSTANT_FILL_CALLS: Set[str] = {"fillna", "nan_to_num"}


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


def _is_numeric_literal(node: ast.AST) -> bool:
    """True for a numeric literal, including a signed one (``-1``, ``+0.0``)."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_numeric_literal(node.operand)
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def _constant_fill_issue(node: ast.Call) -> Optional[str]:
    """Describe the constant fill in ``node``, or ``None`` if it is not one.

    Matches on the called name only, so ``s.fillna(0)``, ``np.nan_to_num(s)`` and a
    bare ``nan_to_num(s)`` are all caught.  Deliberately *not* type-aware: an alpha
    only ever holds pandas/numpy objects, so a false positive would need a
    user-defined ``fillna``, which the one-function contract already forbids.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return None

    if name not in CONSTANT_FILL_CALLS:
        return None

    if name == "nan_to_num":
        # Substitutes 0.0 for NaN by default and has no non-constant form.
        return (
            "nan_to_num() replaces missing values with a constant. Leave them NaN: "
            "the NaN-ratio and tie-mass checks exist to see them"
        )

    # fillna: only a literal is a problem. fillna(method='ffill') and
    # fillna(some_series) both propagate information rather than inventing it.
    filled = next((a for a in node.args if not isinstance(a, ast.Starred)), None)
    if filled is None:
        filled = next((kw.value for kw in node.keywords if kw.arg == "value"), None)
    if filled is None or not _is_numeric_literal(filled):
        return None
    return (
        "fillna() with a constant hides missing data: it drives the measured NaN "
        "ratio to zero, and it puts a stock with no history on the same value -- "
        "and therefore the same rank -- as one genuinely reading that number. "
        "Leave missing values as NaN"
    )


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

    # ----------------------------------------------------------- missing data
    # Deduplicated: a factor that fills three intermediates should say so once.
    seen_fills: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        issue = _constant_fill_issue(node)
        if issue is not None and issue not in seen_fills:
            seen_fills.add(issue)
            issues.append(issue)

    return AuditResult(
        ok=not issues,
        issues=issues,
        function_name=fn_name,
        has_docstring=has_doc,
        imports=sorted(set(imports)),
    )
