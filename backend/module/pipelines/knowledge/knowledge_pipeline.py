from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from backend.module.logging import LoggingInterceptor
from backend.module.providers.embeddings import EmbeddingsClientBundle, get_embeddings_client_bundle

logger = LoggingInterceptor("pipelines_knowledge")


class KnowledgeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_name: str
    query: str
    answer: str
    user: str
    file_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeIngestResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    collection_name: str
    point_id: Optional[str]
    inserted: bool
    reused: bool
    similarity: Optional[float]
    output_path: Path
    info: Dict[str, Any] = Field(default_factory=dict)


async def _ensure_collection(client: AsyncQdrantClient, collection_name: str, vector_size: int) -> None:
    exists = await client.collection_exists(collection_name=collection_name)
    if not exists:
        logger.info("Creating knowledge collection", collection=collection_name, vector_size=vector_size)
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return
    info = await client.get_collection(collection_name=collection_name)
    cfg = info.config.params.vectors
    existing_size = cfg.size if hasattr(cfg, "size") else None
    if existing_size and existing_size != vector_size:
        raise ValueError(
            f"Collection '{collection_name}' vector size {existing_size} does not match required {vector_size}"
        )


async def _embed_text(text: str, bundle: EmbeddingsClientBundle) -> list[float]:
    response = await bundle.client.embeddings.create(model=bundle.model, input=[text])
    vector = response.data[0].embedding
    if not vector:
        raise ValueError("Empty embedding received for knowledge entry")
    return list(vector)


async def _find_similarity(
    client: AsyncQdrantClient, collection_name: str, vector: list[float], threshold: float
) -> tuple[Optional[str], Optional[float]]:
    try:
        result = await client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=1,
            score_threshold=threshold,
            with_payload=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Similarity query failed", error=str(exc))
        return None, None
    if not result.points:
        return None, None
    point = result.points[0]
    return str(point.id), float(point.score)


async def ingest_knowledge_entry(
    entry: KnowledgeEntry,
    *,
    qdrant_path: Path,
    output_path: Path,
    similarity_threshold: float = 0.95,
    force: bool = False,
    provider: Optional[str] = None,
) -> KnowledgeIngestResult:
    """
    Insert or guard knowledge entries in a local Qdrant collection with similarity checks.
    """
    qdrant_path.mkdir(parents=True, exist_ok=True)
    bundle = get_embeddings_client_bundle(provider=provider)
    client = AsyncQdrantClient(path=str(qdrant_path))

    text = f"Q: {entry.query}\nA: {entry.answer}"
    try:
        vector = await _embed_text(text, bundle)
        await _ensure_collection(client, entry.collection_name, len(vector))
        existing_id, existing_score = await _find_similarity(client, entry.collection_name, vector, similarity_threshold)

        if existing_id and not force:
            logger.info(
                "Skipping knowledge insert due to similarity guard",
                collection=entry.collection_name,
                point_id=existing_id,
                score=existing_score,
            )
            return KnowledgeIngestResult(
                collection_name=entry.collection_name,
                point_id=existing_id,
                inserted=False,
                reused=True,
                similarity=existing_score,
                output_path=output_path,
                info={"reason": "similarity_guard"},
            )

        if existing_id and force:
            await client.delete(entry.collection_name, points_selector=[existing_id], wait=True)
            logger.info(
                "Removed existing similar knowledge entry", collection=entry.collection_name, point_id=existing_id
            )

        point_id = str(uuid4())
        payload = {
            "query": entry.query,
            "answer": entry.answer,
            "user": entry.user,
            "file_id": entry.file_id,
            **(entry.metadata or {}),
        }
        upload_method = client.upload_points
        if inspect.iscoroutinefunction(upload_method):
            await upload_method(
                collection_name=entry.collection_name,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                wait=True,
            )
        else:
            await asyncio.to_thread(
                upload_method,
                collection_name=entry.collection_name,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                wait=True,
            )

        output_payload = {
            "collection": entry.collection_name,
            "point_id": point_id,
            "user": entry.user,
            "file_id": entry.file_id,
            "query": entry.query,
            "answer": entry.answer,
            "metadata": entry.metadata,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(output_path.write_text, json_dumps(output_payload), "utf-8")
        logger.info("Knowledge entry ingested", collection=entry.collection_name, point_id=point_id)
        return KnowledgeIngestResult(
            collection_name=entry.collection_name,
            point_id=point_id,
            inserted=True,
            reused=False,
            similarity=existing_score,
            output_path=output_path,
            info={"replaced": bool(existing_id)},
        )
    finally:
        try:
            await client.close()
        except Exception:
            pass
        try:
            await bundle.client.close()
        except Exception:
            pass


def json_dumps(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = ["KnowledgeEntry", "KnowledgeIngestResult", "ingest_knowledge_entry"]
