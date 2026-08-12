"""
Judging modules for Famou 2.0.

Base class:
- JudgeModule: Inherit to create custom code review/feedback modules

Built-in implementations (import from files directly):
- llm_judge.LLMJudge: LLM-based code review and feedback
"""

from famou.modules.judge.base import JudgeModule

__all__ = ["JudgeModule"]
