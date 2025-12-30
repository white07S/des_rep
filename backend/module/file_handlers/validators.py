from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.module.file_handlers import storage
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("file_handlers_validators")


class UploadType(str, Enum):
    PDF = "pdf"
    CSV = "csv"
    SQLITE = "sqlite"


ALLOWED_EXTENSIONS = {
    UploadType.PDF: {".pdf"},
    UploadType.CSV: {".csv"},
    UploadType.SQLITE: {".sqlite"},
}


class UploadRequest(BaseModel):
    """User-submitted upload payload."""

    model_config = ConfigDict(extra="forbid")

    user: str
    business_context: str
    upload_type: UploadType
    source_path: Path
    schema_sql_path: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_paths(self) -> "UploadRequest":
        resolved = self.source_path.expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Source file not found: {resolved}")
        self.source_path = resolved

        allowed = ALLOWED_EXTENSIONS[self.upload_type]
        if resolved.suffix.lower() not in allowed:
            raise ValueError(f"Invalid extension {resolved.suffix}; allowed: {allowed}")

        if self.upload_type is UploadType.SQLITE:
            if not self.schema_sql_path:
                raise ValueError("schema_sql_path is required for sqlite uploads")
            schema_path = self.schema_sql_path.expanduser().resolve()
            if not schema_path.exists() or not schema_path.is_file():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")
            if schema_path.suffix.lower() != ".sql":
                raise ValueError("schema_sql_path must point to a .sql file")
            self.schema_sql_path = schema_path
        return self


class UploadValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, json_encoders={Path: str})

    file_id: str
    file_hash: str
    original_path: Path
    artifact_dir: Path
    original_filename: str
    file_size: int
    schema_path: Optional[Path] = None


def compute_upload_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_upload(request: UploadRequest, *, precomputed_hash: Optional[str] = None) -> UploadValidationResult:
    """Validate upload, stage into originals, and return normalized identifiers."""
    file_hash = precomputed_hash or compute_upload_hash(request.source_path)
    file_id = f"{file_hash[:12]}-{uuid4().hex}"
    logger.info("Computed file identifier", file_hash=file_hash, file_id=file_id)

    original_path = storage.get_original_file_path(request.user, file_id, request.source_path.suffix)
    storage.ensure_directory(original_path.parent)
    storage.atomic_copy(request.source_path, original_path)

    schema_target: Optional[Path] = None
    if request.upload_type is UploadType.SQLITE and request.schema_sql_path:
        schema_target = storage.resolve_original_path(request.user, file_id, "schema.sql")
        storage.atomic_copy(request.schema_sql_path, schema_target)

    artifact_dir = storage.get_artifacts_dir(request.user, file_id)
    storage.ensure_directory(artifact_dir)

    return UploadValidationResult(
        file_id=file_id,
        file_hash=file_hash,
        original_path=original_path,
        artifact_dir=artifact_dir,
        original_filename=request.source_path.name,
        file_size=request.source_path.stat().st_size,
        schema_path=schema_target,
    )


__all__ = ["UploadRequest", "UploadValidationResult", "UploadType", "prepare_upload", "compute_upload_hash"]
