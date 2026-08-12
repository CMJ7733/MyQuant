"""
Centralized prompt registry for Famou 2.0.

Provides template-based prompt management with Jinja2 for:
- Easy experimentation with prompt variations
- Version control of prompts
- Separation of concerns (prompts vs code)
- Reusability across modules
"""

from pathlib import Path
from typing import Any, Dict, Optional
import jinja2
from numbers import Number, Integral


class PromptRegistry:
    """
    Centralized prompt template management.

    Uses Jinja2 templates to generate prompts dynamically based on context.
    Templates are stored in famou/prompts/templates/ and organized by category.

    Features:
    - Template-based prompts with variable substitution
    - Template inheritance and includes
    - Custom filters for formatting
    - Runtime registration of custom templates
    - Template listing and discovery

    Example:
        >>> registry = PromptRegistry()
        >>> prompt = registry.get(
        ...     "generation/mutation.txt",
        ...     language="python",
        ...     parent_code="def foo(): pass",
        ...     user_prompt="Improve this function"
        ... )
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize prompt registry.

        Args:
            templates_dir: Path to templates directory (default: famou/prompts/templates/)
        """
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"

        # Create Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined,  # Fail on undefined variables
        )

        # Add custom filters
        self._register_filters()

        # Cache for custom templates
        self._custom_templates: Dict[str, str] = {}

    def _register_filters(self):
        """Register custom Jinja2 filters for prompt formatting."""

        def format_score(score: Optional[float], precision: int = 4) -> str:
            """Format score for display."""
            if score is None:
                return "N/A"
            return f"{score:.{precision}f}"
        
        def format_number(value: Any, precision: int = 4) -> str:
            """Format number with given precision."""
            if isinstance(value, Number) and not isinstance(value, Integral):
                return f"{value:.{precision}f}"
            return str(value)

        def format_metrics(metrics: Optional[Dict[str, Any]]) -> str:
            """Format metrics dict as readable string."""
            if not metrics:
                return "No metrics"
            items = [f"{k}: {format_number(v)}" for k, v in metrics.items()]
            return "\n".join(items)

        self.env.filters["format_score"] = format_score
        self.env.filters["format_metrics"] = format_metrics

    def get(self, template_name: str, **kwargs) -> str:
        """
        Get rendered prompt template.

        Args:
            template_name: Template path relative to templates_dir (e.g., "generation/mutation.txt")
            **kwargs: Template variables for rendering

        Returns:
            Rendered prompt string

        Raises:
            jinja2.TemplateNotFound: If template doesn't exist
            jinja2.UndefinedError: If required variable is missing

        Example:
            >>> prompt = registry.get(
            ...     "generation/mutation.txt",
            ...     language="python",
            ...     parent_code="def fibonacci(n): return n",
            ...     parent_score=0.5,
            ...     user_prompt="Optimize this function"
            ... )
        """
        # Check custom templates first
        if template_name in self._custom_templates:
            template = jinja2.Template(
                self._custom_templates[template_name],
                undefined=jinja2.StrictUndefined
            )
            return template.render(**kwargs)

        # Provide safe defaults for generation-mode flags so templates that
        # reference these variables don't raise StrictUndefined errors when
        # called by modules that haven't been updated yet.
        kwargs.setdefault("diff_write_mode", False)
        kwargs.setdefault("has_evolve_block", False)

        # Load from file
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

    def register_custom(self, name: str, template: str):
        """
        Register custom prompt template at runtime.

        Useful for experiments or user-defined prompts that don't need to be in files.

        Args:
            name: Template name (use same format as file paths, e.g., "custom/my_prompt.txt")
            template: Jinja2 template string

        Example:
            >>> registry.register_custom(
            ...     "custom/simple.txt",
            ...     "Improve this {{ language }} code:\n{{ code }}"
            ... )
        """
        self._custom_templates[name] = template

    def list_templates(self, pattern: Optional[str] = None) -> list[str]:
        """
        List all available templates.

        Args:
            pattern: Optional glob pattern to filter templates (e.g., "generation/*")

        Returns:
            List of template names

        Example:
            >>> registry.list_templates("generation/*")
            ['generation/mutation.txt', 'generation/crossover.txt', ...]
        """
        all_templates = self.env.list_templates()

        if pattern:
            import fnmatch
            return [t for t in all_templates if fnmatch.fnmatch(t, pattern)]

        return list(all_templates)

    def exists(self, template_name: str) -> bool:
        """
        Check if template exists.

        Args:
            template_name: Template name to check

        Returns:
            True if template exists (in files or custom registry)
        """
        if template_name in self._custom_templates:
            return True

        try:
            self.env.get_template(template_name)
            return True
        except jinja2.TemplateNotFound:
            return False

    def get_template_path(self, template_name: str) -> Optional[Path]:
        """
        Get file path for template.

        Args:
            template_name: Template name

        Returns:
            Path to template file, or None if it's a custom template
        """
        if template_name in self._custom_templates:
            return None

        return self.templates_dir / template_name

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"PromptRegistry("
            f"templates_dir={self.templates_dir}, "
            f"templates={len(self.list_templates())}, "
            f"custom={len(self._custom_templates)})"
        )


# Global registry instance (singleton pattern)
prompt_registry = PromptRegistry()
