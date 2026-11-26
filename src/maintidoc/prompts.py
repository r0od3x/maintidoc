"""Versioned prompt loading.

Prompts live in `prompts/<name>_<version>.yaml` and are tracked in Git, as the
brief requires. Loading returns the text *and* its SHA-256, so a run can record
exactly which prompt text produced it — a version tag alone would not catch an
edit made in place.

Templates use `$placeholder` (`string.Template`) rather than `str.format`,
because evidence text routinely contains braces from tables and JSON, which
would break `.format()`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Self

import yaml

from maintidoc.config import get_settings
from maintidoc.contracts import PromptVersion

REQUIRED_KEYS = {"name", "version", "system", "user"}


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    version: str
    sha256: str
    system: str
    user: str
    metadata: dict[str, Any]

    @classmethod
    def from_file(cls, path: Path) -> Self:
        raw = path.read_bytes()
        data = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path.name} : le prompt doit être un mapping YAML")
        missing = REQUIRED_KEYS - data.keys()
        if missing:
            raise ValueError(f"{path.name} : clés manquantes {sorted(missing)}")
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            sha256=hashlib.sha256(raw).hexdigest(),
            system=str(data["system"]),
            user=str(data["user"]),
            metadata={k: v for k, v in data.items() if k not in REQUIRED_KEYS},
        )

    def render_user(self, **values: Any) -> str:
        """Strict substitution: an unknown or missing placeholder is an error,
        which turns a silently half-filled prompt into a loud failure."""
        return Template(self.user).substitute(**values)

    @property
    def version_record(self) -> PromptVersion:
        return PromptVersion(name=self.name, version=self.version, sha256=self.sha256)


def prompt_path(name: str, version: str = "v1", root: Path | None = None) -> Path:
    base = root or get_settings().prompts_dir
    return base / f"{name}_{version}.yaml"


def load_prompt(name: str, version: str = "v1", root: Path | None = None) -> PromptTemplate:
    path = prompt_path(name, version, root)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt introuvable : {path}")
    template = PromptTemplate.from_file(path)
    if template.name != name or template.version != version:
        raise ValueError(
            f"{path.name} déclare {template.name}/{template.version}, "
            f"attendu {name}/{version}"
        )
    return template


def load_all(root: Path | None = None) -> dict[str, PromptTemplate]:
    """Every prompt in the directory, keyed by name. Used to fingerprint a run."""
    base = root or get_settings().prompts_dir
    templates: dict[str, PromptTemplate] = {}
    for path in sorted(base.glob("*.yaml")):
        template = PromptTemplate.from_file(path)
        templates[template.name] = template
    return templates
