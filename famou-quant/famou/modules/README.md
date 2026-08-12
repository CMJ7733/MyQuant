# Module Contribution Guide

Keep modules small, single-purpose, and easy to compose in a rollout.

Preferred layout:
- Base classes in `famou/modules/<type>/base.py`
- Concrete modules in `famou/modules/<type>/<name>.py`
- Names: file snake_case, class CamelCase

Steps:
1) Pick a module type: select, generate, evaluate, judge, feature, population.
2) Subclass the base class and implement `validate_input`, `execute`, and `validate_output`.
3) Use dependency mixins (`RequiresLLM`, `RequiresEnv`, `RequiresEmbedding`) when needed.
4) Export in `famou/modules/<type>/__init__.py` and `famou/modules/__init__.py`.
5) Add a small test in `tests/test_modules/` when feasible.

If you need to work on a legacy branch with flat module files, add your class
to `famou/modules/<type>.py` and keep the export steps the same.
