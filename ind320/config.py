import os
from pathlib import Path

from dotenv import dotenv_values

# Resolve private settings from the project root after moving into the package.
ROOT = Path(__file__).resolve().parents[1]


def get_setting(name: str, default: str = "") -> str:
    """Prefer environment, then Cloud secrets, then local dotenv settings.

    Keep credentials in private settings, never in source files or output.
    """
    if name in os.environ:
        return os.environ[name]

    import streamlit as st

    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass

    # Read local settings without changing the process environment.
    local = dotenv_values(ROOT / ".env", interpolate=False)
    return str(local.get(name) or default)
