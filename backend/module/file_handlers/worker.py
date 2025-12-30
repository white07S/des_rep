from __future__ import annotations

from typing import Dict, Iterable, Optional

from saq.queue.base import Queue
from backend.module.config_handler import settings
from backend.module.file_handlers.storage import ensure_directory
from backend.module.file_handlers.sqlite_queue import SQLiteQueue
from saq import Worker
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("file_handlers_worker")

_QUEUE: Optional[Queue] = None


def get_queue() -> Queue:
    """Return the shared SAQ queue backed by SQLite in DATA_ROOT."""
    global _QUEUE
    if _QUEUE is None:
        jobs_db = settings.storage.data_root / "jobs.db"
        ensure_directory(jobs_db.parent)
        _QUEUE = SQLiteQueue(jobs_db)
        logger.info("Queue initialized", path=str(jobs_db))
    return _QUEUE


async def start_worker(functions: Iterable, *, concurrency: int = 2, context: Optional[Dict] = None) -> Worker:
    """
    Start a SAQ worker in-process.

    Args:
        functions: Iterable of task callables registered with the worker.
        concurrency: Number of concurrent tasks to process.
        context: Optional base context injected into each task's ctx.
    """
    queue = get_queue()
    worker = Worker(queue, functions=list(functions), concurrency=concurrency, context=context or {})
    await worker.start()
    logger.info("Worker started", concurrency=concurrency, functions=[f.__name__ for f in functions])
    return worker


async def stop_worker(worker: Worker) -> None:
    """Stop a running worker."""
    await worker.stop()
    logger.info("Worker stopped")


async def start_default_worker(concurrency: int = 2) -> Worker:
    """Start a worker with the default dispatcher task set."""
    from backend.module.file_handlers.dispatchers import process_csv, process_knowledge, process_pdf, process_sqlite

    return await start_worker(
        functions=[process_pdf, process_csv, process_sqlite, process_knowledge],
        concurrency=concurrency,
    )


__all__ = ["get_queue", "start_worker", "stop_worker", "start_default_worker"]
