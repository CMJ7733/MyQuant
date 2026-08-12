"""Multi-Agent Quality Checker and its deterministic gates (§3.3, Appendix A.3)."""

from cogalpha.quality.audit import AuditResult, audit_code  # noqa: F401
from cogalpha.quality.numeric import NumericReport, check_numeric  # noqa: F401
from cogalpha.quality.leakage import (  # noqa: F401
    LeakageReport,
    build_report,
    determinism_check,
    scan_lookahead,
    truncation_probe,
)
from cogalpha.quality.sandbox import (  # noqa: F401
    ExecOutcome,
    SandboxError,
    SandboxRunner,
    apply_alpha,
    compile_alpha,
)
from cogalpha.quality.checker import CheckerStats, QualityChecker  # noqa: F401

__all__ = [
    "AuditResult",
    "audit_code",
    "NumericReport",
    "check_numeric",
    "LeakageReport",
    "build_report",
    "determinism_check",
    "scan_lookahead",
    "truncation_probe",
    "ExecOutcome",
    "SandboxError",
    "SandboxRunner",
    "apply_alpha",
    "compile_alpha",
    "CheckerStats",
    "QualityChecker",
]
