"""Read private settings without printing credentials or requiring databases."""

import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent


class ConfigurationError(RuntimeError):
    """A required private setting is missing or invalid."""


def get_setting(name: str, default: str = "") -> str:
    """Prefer environment, then Cloud secrets, then local dotenv settings.

    The existing Atlas file is supported for this workspace. Values are never
    copied into tracked files. Cloud secrets use top-level names from the example.
    """
    if name in os.environ:
        return os.environ[name]

    import streamlit as st

    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass

    # .env overrides the older Atlas-specific file, without modifying os.environ.
    local = {
        **dotenv_values(ROOT / "atlas-credentials.env", interpolate=False),
        **dotenv_values(ROOT / ".env", interpolate=False),
    }
    return str(local.get(name) or default)


def require_setting(name: str) -> str:
    """Report a missing setting by name, never by its sensitive value."""
    value = get_setting(name)
    if not value:
        raise ConfigurationError(f"Set {name} in .env or Streamlit secrets.")
    return value
