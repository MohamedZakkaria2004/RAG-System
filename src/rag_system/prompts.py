from __future__ import annotations

from pathlib import Path

import yaml


class PromptRegistry:
    def __init__(self, prompt_dir: str | Path = "prompts") -> None:
        self.prompt_dir = Path(prompt_dir)

    def load(self, name: str, version: str = "v1") -> str:
        path = self.prompt_dir / f"{name}.{version}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        template = payload.get("template")
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"Prompt file has no template: {path}")
        return template
