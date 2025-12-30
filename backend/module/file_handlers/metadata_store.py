from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from backend.module.file_handlers.storage import atomic_write_text, ensure_directory
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("file_handlers_metadata_store")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, json_encoders={Path: str})

    file_id: str
    user: str
    business_context: str
    file_type: str
    original_filename: str
    original_path: Path
    artifact_path: Path
    file_hash: str
    file_size: int
    created_at: datetime
    updated_at: datetime
    extra: Dict[str, Any] = {}


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, json_encoders={Path: str})

    job_id: str
    file_id: Optional[str] = None
    user: Optional[str] = None
    job_type: str
    state: str
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    artifact_path: Optional[Path] = None
    extra: Dict[str, Any] = {}


class KnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, json_encoders={Path: str})

    knowledge_id: str
    user: str
    collection_name: str
    query: str
    answer_path: Path
    similarity: Optional[float] = None
    forced: bool = False
    file_id: Optional[str] = None
    created_at: datetime
    extra: Dict[str, Any] = {}


@dataclass
class _Tables:
    metadata: MetaData
    files: Table
    jobs: Table
    knowledge: Table


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"Cannot coerce {value!r} to datetime")


class MetadataStore:
    """Dual-write metadata store (SQLite + JSON snapshots)."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = ensure_directory(base_dir)
        self.db_path = self.base_dir / "metadata.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self.tables = self._build_tables()
        self._ensure_schema()
        self._json_paths = {
            "files": self.base_dir / "files.json",
            "jobs": self.base_dir / "jobs.json",
            "knowledge": self.base_dir / "knowledge.json",
        }
        logger.info("Metadata store initialized", db=str(self.db_path))

    def _build_tables(self) -> _Tables:
        metadata = MetaData()
        files = Table(
            "files",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("file_id", String, unique=True, nullable=False),
            Column("user", String, nullable=False),
            Column("business_context", Text, nullable=False),
            Column("file_type", String, nullable=False),
            Column("original_filename", String, nullable=False),
            Column("original_path", String, nullable=False),
            Column("artifact_path", String, nullable=False),
            Column("file_hash", String, nullable=False),
            Column("file_size", Integer, nullable=False),
            Column("extra", JSON, nullable=False, default={}),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
        jobs = Table(
            "jobs",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("job_id", String, unique=True, nullable=False),
            Column("file_id", String, nullable=True),
            Column("user", String, nullable=True),
            Column("job_type", String, nullable=False),
            Column("state", String, nullable=False),
            Column("error", Text, nullable=True),
            Column("artifact_path", String, nullable=True),
            Column("extra", JSON, nullable=False, default={}),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("started_at", DateTime(timezone=True), nullable=True),
            Column("finished_at", DateTime(timezone=True), nullable=True),
        )
        knowledge = Table(
            "knowledge",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("knowledge_id", String, unique=True, nullable=False),
            Column("user", String, nullable=False),
            Column("collection_name", String, nullable=False),
            Column("query", Text, nullable=False),
            Column("answer_path", String, nullable=False),
            Column("similarity", Float, nullable=True),
            Column("forced", Boolean, nullable=False, default=False),
            Column("file_id", String, nullable=True),
            Column("extra", JSON, nullable=False, default={}),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )
        return _Tables(metadata=metadata, files=files, jobs=jobs, knowledge=knowledge)

    def _ensure_schema(self) -> None:
        self.tables.metadata.create_all(self.engine)

    def _write_snapshot(self, rows: Iterable[Dict[str, Any]], key: str, path: Path) -> None:
        data: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            item = dict(row)
            value = item.pop(key)
            data[value] = item
        payload = json.dumps(data, default=str, indent=2)
        atomic_write_text(path, payload)

    def _fetch_rows(self, table: Table) -> Iterable[Dict[str, Any]]:
        with self.engine.begin() as conn:
            result = conn.execute(select(table)).mappings().all()
        return [dict(row) for row in result]

    def _row_to_file_record(self, row: Dict[str, Any]) -> FileRecord:
        return FileRecord(
            file_id=row["file_id"],
            user=row["user"],
            business_context=row["business_context"],
            file_type=row["file_type"],
            original_filename=row["original_filename"],
            original_path=Path(row["original_path"]),
            artifact_path=Path(row["artifact_path"]),
            file_hash=row["file_hash"],
            file_size=row["file_size"],
            created_at=_coerce_datetime(row["created_at"]),
            updated_at=_coerce_datetime(row["updated_at"]),
            extra=row.get("extra", {}) or {},
        )

    def get_file_by_hash(self, *, user: str, file_hash: str, file_type: str) -> Optional[FileRecord]:
        stmt = select(self.tables.files).where(
            self.tables.files.c.user == user,
            self.tables.files.c.file_hash == file_hash,
            self.tables.files.c.file_type == file_type,
        )
        with self.engine.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if not row:
            return None
        return self._row_to_file_record(dict(row))

    def upsert_file(self, record: FileRecord) -> None:
        now = _utc_now()
        payload = {
            **record.model_dump(exclude={"created_at", "updated_at"}),
            "created_at": record.created_at,
            "updated_at": now,
        }
        payload["original_path"] = str(payload["original_path"])
        payload["artifact_path"] = str(payload["artifact_path"])
        stmt = sqlite_insert(self.tables.files).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.tables.files.c.file_id],
            set_={**payload, "created_at": self.tables.files.c.created_at},
        )
        self._execute_and_snapshot(stmt, "files", key="file_id")
        logger.info("File metadata upserted", file_id=record.file_id, user=record.user)

    def upsert_job(self, record: JobRecord) -> None:
        now = _utc_now()
        payload = {
            **record.model_dump(exclude={"created_at", "updated_at"}),
            "created_at": record.created_at,
            "updated_at": now,
        }
        if payload.get("artifact_path"):
            payload["artifact_path"] = str(payload["artifact_path"])
        stmt = sqlite_insert(self.tables.jobs).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.tables.jobs.c.job_id],
            set_={
                **payload,
                "created_at": self.tables.jobs.c.created_at,
            },
        )
        self._execute_and_snapshot(stmt, "jobs", key="job_id")
        logger.info("Job metadata upserted", job_id=record.job_id, state=record.state)

    def upsert_knowledge(self, record: KnowledgeRecord) -> None:
        payload = record.model_dump()
        payload["answer_path"] = str(payload["answer_path"])
        stmt = sqlite_insert(self.tables.knowledge).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.tables.knowledge.c.knowledge_id],
            set_=payload,
        )
        self._execute_and_snapshot(stmt, "knowledge", key="knowledge_id")
        logger.info(
            "Knowledge metadata upserted",
            knowledge_id=record.knowledge_id,
            collection=record.collection_name,
        )

    def _execute_and_snapshot(self, statement, table_key: str, *, key: str) -> None:
        table_map = {
            "files": self.tables.files,
            "jobs": self.tables.jobs,
            "knowledge": self.tables.knowledge,
        }
        table = table_map[table_key]
        with self.engine.begin() as conn:
            conn.execute(statement)
        rows = self._fetch_rows(table)
        self._write_snapshot(rows, key=key, path=self._json_paths[table_key])


def create_metadata_store(data_root: Path) -> MetadataStore:
    return MetadataStore(base_dir=data_root / "metadata")


__all__ = ["MetadataStore", "FileRecord", "JobRecord", "KnowledgeRecord", "create_metadata_store"]
