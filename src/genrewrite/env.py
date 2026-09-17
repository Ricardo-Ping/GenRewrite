"""Loader for the local ``.env`` configuration file.

The configuration lives in a plain-text dotenv file (``.env``) at the
project root — not in any Python file. It is listed in ``.gitignore``
and must never be committed.

Supported variable names (checked in this order):
    GENREWRITE_API_BASE / OPENAI_API_BASE / OPENAI_BASE_URL
    GENREWRITE_API_KEY  / OPENAI_API_KEY
    GENREWRITE_MODEL    / OPENAI_MODEL

Usage::

    from genrewrite.env import get_config

    config = get_config()   # {"base_url": ..., "api_key": ..., "model": ...}
"""

import os
from pathlib import Path

# Project root: src/genrewrite/env.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# Config item -> env var names tried in order (real env vars win first).
_ENV_KEYS = {
    "base_url": ("GENREWRITE_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"),
    "api_key": ("GENREWRITE_API_KEY", "OPENAI_API_KEY"),
    "model": ("GENREWRITE_MODEL", "OPENAI_MODEL"),
}

_DEFAULTS = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "",
    "model": "gpt-4o-mini",
}


def _parse_env_file(path: Path) -> dict:
    """Parse a simple ``KEY=VALUE`` dotenv file.

    Supports blank lines, ``#`` comments, and surrounding quotes.
    """
    values = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def get_config(env_file=None) -> dict:
    """Return LLM API config: ``base_url``, ``api_key``, ``model``.

    Lookup order: real environment variables -> ``.env`` file -> defaults.
    Raises ``RuntimeError`` when the API key is missing.
    """
    file_values = _parse_env_file(Path(env_file) if env_file else DEFAULT_ENV_FILE)
    config = {}
    for name, keys in _ENV_KEYS.items():
        value = None
        for candidate in keys:
            value = os.environ.get(candidate)
            if value:
                break
        if not value:
            value = file_values.get(candidate, None)
        # fall back through all key aliases inside the file too
        if not value:
            for candidate in keys:
                if file_values.get(candidate):
                    value = file_values[candidate]
                    break
        config[name] = value or _DEFAULTS[name]
    if not config["api_key"]:
        raise RuntimeError(
            "Missing API key. Set GENREWRITE_API_KEY (or OPENAI_API_KEY) as "
            "an environment variable, or edit the .env file at the project "
            "root."
        )
    return config
