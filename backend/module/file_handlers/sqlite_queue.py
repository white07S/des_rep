from __future__ import annotations

import asyncio
import json
import sqlite3
import typing as t
from pathlib import Path
from time import time

from saq.job import Job, Status
from saq.queue.base import JobError, Queue
from saq.types import CountKind, QueueInfo, WorkerInfo

from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("file_handlers_sqlite_queue")


class SQLiteQueue(Queue):
    """
    Lightweight SQLite-backed SAQ queue for local processing.
    Stores serialized Job payloads and minimal status metadata.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        super().__init__(name="file_handler_queue", dump=json.dumps, load=json.loads)
        logger.info("SQLite queue initialized", path=str(self.db_path))

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                queued REAL DEFAULT 0,
                started REAL DEFAULT 0,
                completed REAL DEFAULT 0
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        self._conn.commit()

    async def _run(self, fn: t.Callable[[], t.Any]) -> t.Any:
        async with self._lock:
            return await asyncio.to_thread(fn)

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        payload = self.deserialize(row["payload"])
        assert payload is not None
        job = t.cast(Job, payload)
        job.queue = self
        return job

    async def _enqueue(self, job: Job) -> Job | None:
        def op() -> Job:
            payload = self.serialize(job)
            cur = self._conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO jobs (key, payload, status, queued, started, completed) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    job.key,
                    payload,
                    job.status.value,
                    job.queued,
                    job.started,
                    job.completed,
                ),
            )
            self._conn.commit()
            return job

        return await self._run(op)

    async def dequeue(self, timeout: float = 0.0, poll_interval: float = 0.0) -> Job | None:
        deadline = time() + timeout if timeout else None

        async def fetch_one() -> Job | None:
            def op() -> Job | None:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT payload FROM jobs
                    WHERE status IN (?, ?)
                    ORDER BY queued ASC
                    LIMIT 1
                    """,
                    (Status.NEW.value, Status.QUEUED.value),
                )
                row = cur.fetchone()
                if not row:
                    return None
                job = self._row_to_job(row)
                # Mark as active to avoid double dequeues
                job.status = Status.ACTIVE
                job.started = int(time())
                payload = self.serialize(job)
                cur.execute(
                    "UPDATE jobs SET payload = ?, status = ?, started = ? WHERE key = ?",
                    (payload, job.status.value, job.started, job.key),
                )
                self._conn.commit()
                return job

            return await self._run(op)

        while True:
            job = await fetch_one()
            if job:
                return job
            if deadline and time() > deadline:
                return None
            await asyncio.sleep(poll_interval or 0.25)

    async def _update(self, job: Job, status: Status | None = None, **kwargs: t.Any) -> None:
        if status:
            job.status = status
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        await self._persist(job)

    async def job(self, job_key: str) -> Job | None:
        def op() -> Job | None:
            cur = self._conn.cursor()
            cur.execute("SELECT payload FROM jobs WHERE key = ?", (job_key,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_job(row)

        return await self._run(op)

    async def jobs(self, job_keys: t.Iterable[str]) -> list[Job | None]:
        return [await self.job(key) for key in job_keys]

    def iter_jobs(
        self,
        statuses: list[Status] = list(Status),
        batch_size: int = 100,
    ) -> t.AsyncIterator[Job]:
        status_values = [s.value for s in statuses]

        async def generator() -> t.AsyncIterator[Job]:
            offset = 0
            while True:
                def op() -> list[sqlite3.Row]:
                    cur = self._conn.cursor()
                    cur.execute(
                        """
                        SELECT payload FROM jobs
                        WHERE status IN ({placeholders})
                        LIMIT ? OFFSET ?
                        """.format(
                            placeholders=",".join("?" for _ in status_values)
                        ),
                        (*status_values, batch_size, offset),
                    )
                    return cur.fetchall()

                rows = await self._run(op)
                if not rows:
                    break
                for row in rows:
                    yield self._row_to_job(row)
                offset += batch_size

        return generator()

    async def abort(self, job: Job, error: str, ttl: float = 5) -> None:
        job.status = Status.ABORTED
        job.error = error
        await self._persist(job)

    async def _finish(
        self,
        job: Job,
        status: Status,
        *,
        result: t.Any = None,
        error: str | None = None,
    ) -> None:
        job.status = status
        job.result = result
        job.error = error
        await self._persist(job)

    async def _retry(self, job: Job, error: str | None) -> None:
        job.error = error
        await self._persist(job)

    async def _persist(self, job: Job) -> None:
        def op() -> None:
            payload = self.serialize(job)
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO jobs (key, payload, status, queued, started, completed)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job.key,
                    payload,
                    job.status.value,
                    job.queued,
                    job.started,
                    job.completed,
                ),
            )
            self._conn.commit()

        await self._run(op)

    async def count(self, kind: CountKind) -> int:
        status_filter = {
            "queued": [Status.QUEUED.value, Status.NEW.value],
            "active": [Status.ACTIVE.value],
            "incomplete": [Status.NEW.value, Status.QUEUED.value, Status.ACTIVE.value],
        }[kind]

        def op() -> int:
            cur = self._conn.cursor()
            placeholders = ",".join("?" for _ in status_filter)
            cur.execute(f"SELECT COUNT(*) as c FROM jobs WHERE status IN ({placeholders})", status_filter)
            row = cur.fetchone()
            return int(row["c"] if row else 0)

        return await self._run(op)

    async def info(self, jobs: bool = False, offset: int = 0, limit: int = 10) -> QueueInfo:
        queued = await self.count("queued")
        active = await self.count("active")
        jobs_list: list[dict[str, t.Any]] = []

        if jobs:
            def op() -> list[dict[str, t.Any]]:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT payload FROM jobs
                    WHERE status IN (?, ?)
                    ORDER BY queued ASC
                    LIMIT ? OFFSET ?
                    """,
                    (Status.QUEUED.value, Status.NEW.value, limit, offset),
                )
                rows = cur.fetchall()
                return [json.loads(row["payload"]) for row in rows]

            jobs_list = await self._run(op)

        return {
            "workers": {},  # type: ignore[return-value]
            "name": self.name,
            "queued": queued,
            "active": active,
            "scheduled": 0,
            "jobs": jobs_list,  # type: ignore[return-value]
        }

    async def sweep(self, lock: int = 60, abort: float = 5.0) -> list[str]:
        return []

    async def disconnect(self) -> None:
        def op() -> None:
            try:
                self._conn.close()
            except Exception:
                pass

        await self._run(op)

    async def notify(self, job: Job) -> None:
        # SQLite backend runs single-process; no external notifications required.
        return None

    async def write_worker_info(self, worker_id: str, info: WorkerInfo, ttl: int) -> None:
        return None

    async def _update_job_row(self, job: Job) -> None:
        await self._persist(job)

    def __repr__(self) -> str:
        return f"SQLiteQueue<{self.name}>({self.db_path})"


__all__ = ["SQLiteQueue", "JobError"]
