"""
Prompt management system for Famou 2.0.

Centralized template-based prompt management with:
- Jinja2 templates for flexibility
- Version control integration
- Easy experimentation with prompt variations
- Separation of prompts from code logic
"""

from famou.prompts.registry import PromptRegistry, prompt_registry

__all__ = ["PromptRegistry", "prompt_registry"]
