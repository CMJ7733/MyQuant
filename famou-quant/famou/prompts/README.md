# Prompt Management System

Centralized template-based prompt management for Famou 2.0.

## Overview

All prompts are stored as Jinja2 templates in `famou/prompts/templates/` and managed through the `PromptRegistry`. This provides:

- **Version Control**: Prompts are tracked in git alongside code
- **Easy Experimentation**: Swap templates without changing code
- **Separation of Concerns**: Prompts separate from module logic
- **Reusability**: Share templates across modules
- **Dynamic Rendering**: Jinja2 for flexible variable substitution

## Directory Structure

```
famou/prompts/
├── __init__.py              # Module exports
├── registry.py              # PromptRegistry implementation
├── README.md                # This file
└── templates/               # Jinja2 templates
    ├── generation/
    │   ├── mutation.txt     # Main mutation generation prompt
    │   └── seed.txt         # Seed program generation
    ├── evaluation/
    │   └── judge_feedback.txt  # LLM feedback generation
    └── system/
        └── code_optimizer.txt  # System prompts
```

## Usage

### Basic Usage

```python
from famou.prompts import prompt_registry

# Render a template with variables
prompt = prompt_registry.get(
    "generation/mutation.txt",
    language="python",
    parent_code="def foo(): pass",
    user_prompt="Make this function better",
    parent_score=0.5,
    parent_metrics={"lines": 1},
    error_message=None,
    inspirations=[],
)
```

### In Modules

Modules use the registry to load and render templates:

```python
from famou.prompts import prompt_registry

class MutationGenerate(GenerateModule):
    def build_prompt(self, parent: Program, context: Context) -> str:
        # Load template and render with context
        return prompt_registry.get(
            "generation/mutation.txt",
            language=context.config.get("language", "python"),
            parent_code=parent.code,
            parent_score=parent.combined_score,
            user_prompt=context.user_prompt,
            # ... other variables
        )
```

### Custom Templates

You can override templates in module config:

```python
# Use a different template file
generator = MutationGenerate(
    name="mutation",
    prompt_template="generation/custom_mutation.txt"
)

# Or register a custom template at runtime
prompt_registry.register_custom(
    "custom/my_prompt.txt",
    "Improve this {{ language }} code:\n{{ parent_code }}"
)
```

## Template Format

Templates use Jinja2 syntax with custom filters.

### Available Variables

Templates receive different variables depending on context:

**Generation Templates (`generation/*.txt`):**
- `language`: Programming language (e.g., "python")
- `parent_code`: Parent program's source code
- `parent_score`: Parent's combined score (float or None)
- `parent_metrics`: Parent's evaluation metrics (dict)
- `error_message`: Error from parent (str or None)
- `user_prompt`: User's task description
- `inspirations`: List of high-performing programs (optional)

**Evaluation Templates (`evaluation/*.txt`):**
- `language`: Programming language
- `code`: Code to review
- `user_prompt`: Task description
- `execution_status`: Execution status (str)
- `combined_score`: Program's score (float or None)
- `metrics`: Evaluation metrics (dict)
- `criteria`: Review criteria (list of strings)

### Custom Filters

The registry provides custom Jinja2 filters:

```jinja2
{# Format score with precision #}
Score: {{ parent_score | format_score }}  {# "0.850" or "N/A" #}

{# Format metrics dict as readable string #}
Metrics: {{ metrics | format_metrics }}  {# "lines=10, complexity=5" #}

{# Indent code block #}
{{ code | indent_code(4) }}  {# Indents each line by 4 spaces #}
```

### Example Template

```jinja2
You are an expert {{ language }} programmer.

**Current Code:**
```{{ language }}
{{ parent_code }}
```

{% if parent_score is not none %}
**Current Score:** {{ parent_score | format_score }}
{% endif %}

{% if error_message %}
**Previous Error:**
{{ error_message }}
Please fix this error.
{% endif %}

{% if user_prompt %}
**Requirements:**
{{ user_prompt }}
{% endif %}

Generate an improved version.
```

## Creating New Templates

1. **Create template file** in appropriate directory:
   ```bash
   touch famou/prompts/templates/generation/my_strategy.txt
   ```

2. **Write template** using Jinja2 syntax:
   ```jinja2
   Improve this {{ language }} code:
   {{ parent_code }}

   Focus on {{ focus_area }}.
   ```

3. **Use in module**:
   ```python
   prompt = prompt_registry.get(
       "generation/my_strategy.txt",
       language="python",
       parent_code="...",
       focus_area="performance"
   )
   ```

## Prompt Versioning

Prompts are version-controlled through git:

```bash
# See prompt history
git log famou/prompts/templates/generation/mutation.txt

# Compare prompt versions
git diff HEAD~1 famou/prompts/templates/generation/mutation.txt
```

For experiments, you can:
1. Create a new template file (e.g., `mutation_v2.txt`)
2. Configure module to use it: `prompt_template="generation/mutation_v2.txt"`
3. Compare results

## API Reference

### PromptRegistry

**`get(template_name: str, **kwargs) -> str`**
- Load and render a template with variables
- Raises `TemplateNotFound` if template doesn't exist
- Raises `UndefinedError` if required variable is missing

**`register_custom(name: str, template: str)`**
- Register a template string at runtime
- Useful for experiments or user-defined prompts

**`list_templates(pattern: Optional[str] = None) -> list[str]`**
- List available templates
- Optional glob pattern filter

**`exists(template_name: str) -> bool`**
- Check if template exists

**`get_template_path(template_name: str) -> Optional[Path]`**
- Get file path for template

## Best Practices

1. **Keep prompts DRY**: Use template inheritance for common sections
2. **Version control**: Commit prompt changes with code changes
3. **Document variables**: Comment what each template expects
4. **Test prompts**: Verify templates render correctly
5. **A/B test**: Create variants to compare performance
6. **Track performance**: Note which prompts work best in experiments

## Examples

### Experiment with Prompt Variations

```python
# Run evolution with different prompts
configs = [
    {"prompt_template": "generation/mutation.txt"},
    {"prompt_template": "generation/mutation_aggressive.txt"},
    {"prompt_template": "generation/mutation_conservative.txt"},
]

results = []
for config in configs:
    generator = MutationGenerate(name="mutation", **config)
    rollout = Rollout([TopKSelect(), generator, EvaluateModule(...)])
    result = engine.run_rollout(rollout, context, iteration=0)
    results.append(result)

# Compare which prompt performed best
best = max(results, key=lambda r: r.generated_program.combined_score)
```

### Include Inspirations

```python
# Enable inspirations in generation
generator = MutationGenerate(
    name="mutation",
    include_inspirations=True,  # Adds top-3 programs to prompt
)
```

The template will automatically include inspiration examples if provided.

## Troubleshooting

**Error: `TemplateNotFound`**
- Check template name matches file path
- Verify file exists in `famou/prompts/templates/`

**Error: `UndefinedError: 'variable_name' is undefined`**
- Template requires a variable you didn't provide
- Check template to see what variables it expects
- Provide the missing variable in `get()` call

**IDE shows "prompt_registry is not accessed"**
- False positive - the import is used in method bodies
- Safe to ignore or add `# noqa: F401` if using flake8
