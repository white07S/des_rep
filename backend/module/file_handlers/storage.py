from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("file_handlers_storage")


def _data_root() -> Path:
    """Return the configured data root, ensuring it exists."""
    root = settings.storage.data_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_within_base(base: Path, *parts: str | Path) -> Path:
    """Join paths and assert the result stays within the base directory."""
    target = (base.joinpath(*parts)).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Unsafe path outside data root: {target}")
    return target


def get_originals_dir(user: str, file_id: str) -> Path:
    return _resolve_within_base(_data_root(), "originals", user, file_id)


def get_artifacts_dir(user: str, file_id: str) -> Path:
    return _resolve_within_base(_data_root(), "artifacts", user, file_id)


def get_original_file_path(user: str, file_id: str, extension: str) -> Path:
    ext = extension if extension.startswith(".") else f".{extension}"
    return get_originals_dir(user, file_id) / f"source{ext.lower()}"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(target: Path, writer: Callable[[Path], None]) -> Path:
    ensure_directory(target.parent)
    temp_path = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
    writer(temp_path)
    temp_path.replace(target)
    return target


def atomic_write_bytes(target: Path, content: bytes) -> Path:
    """Write bytes atomically to target path."""
    logger.info("Atomic write bytes", target=str(target))
    return _atomic_write(target, lambda tmp: tmp.write_bytes(content))


def atomic_write_text(target: Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Write text atomically to target path."""
    logger.info("Atomic write text", target=str(target))
    return _atomic_write(target, lambda tmp: tmp.write_text(content, encoding=encoding))


def atomic_copy(source: Path, target: Path) -> Path:
    """Copy a file into target atomically."""
    logger.info("Atomic copy", source=str(source), target=str(target))
    return _atomic_write(target, lambda tmp: shutil.copyfile(source, tmp))


def ensure_path_is_under_data_root(path: Path) -> Path:
    """Validate that a path is within the configured data root."""
    base = _data_root()
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"Path {resolved} escapes data root {base}")
    return resolved


def resolve_artifact_path(user: str, file_id: str, *parts: str | Path) -> Path:
    """Resolve a path within the artifacts directory for the given user/file."""
    return _resolve_within_base(get_artifacts_dir(user, file_id), *parts)


def resolve_original_path(user: str, file_id: str, *parts: str | Path) -> Path:
    """Resolve a path within the originals directory for the given user/file."""
    return _resolve_within_base(get_originals_dir(user, file_id), *parts)
