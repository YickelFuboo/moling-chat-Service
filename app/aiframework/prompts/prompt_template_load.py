import os
import dataclasses
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape


# Initialize Jinja2 environment 
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)

def get_prompt_template(prompt_path: str) -> str:
    """
    Load and return a prompt template using Jinja2.

    Args:
        prompt_path: Name of the prompt template file (without .md extension)

    Returns:
        The template string with proper variable substitution syntax
    """
    try:
        template = env.get_template(f"{prompt_path}.md")
        return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")

def get_prompt_template_with_params(
    prompt_path: str, params: dict = None
) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute

    Returns:
        List of messages with the system prompt as the first message
    """
    try:
        template = env.get_template(f"{prompt_path}.md")
        prompt = template.render(**params)
        return prompt
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_path}: {e}")
