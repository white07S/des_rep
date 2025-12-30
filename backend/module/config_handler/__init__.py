"""
Configuration handler for backend.module.
Loads .env and config.toml with ${env:VAR|default} expansion and exposes typed settings.
"""

from backend.module.config_handler.config import settings  # noqa: F401

