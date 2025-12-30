from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from uuid import uuid4

from pydantic import BaseModel
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal
from qdrant_client.http.models import (
    CollectionInfo,
    Distance,
    PointStruct,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, UniqueConstraint, create_engine, select, update
from sqlalchemy.engine import Engine

from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.docs.document_pipeline_models import ChunkIngestRecord, DocumentIngestResult
from backend.module.providers.embeddings import EmbeddingsClientBundle, get_embeddings_client_bundle

logger = LoggingInterceptor("pipelines_docs_ingest")


class IngestChunk(BaseModel):
    """Normalized chunk ready for ingestion."""

    order: int
    text: str
    original_tag_type: str


class SimilarityMatch(BaseModel):
    """Container for a similarity match in Qdrant."""

    chunk_id: str
    score: float


@dataclass
class _SQLiteMappingStore:
    engine: Engine
    table: Table


def _extract_chunks(markdown: str) -> List[IngestChunk]:
    """
    Extract ordered chunks from <p> and <pai> tags, stripping the tags.
    Chunks are returned in the order they appear in the markdown.
    """
    pattern = re.compile(r"<(?P<tag>p|pai)>(?P<body>.*?)</(?P=tag)>", re.IGNORECASE | re.DOTALL)
    chunks: List[IngestChunk] = []
    for idx, match in enumerate(pattern.finditer(markdown)):
        tag = match.group("tag").lower()
        body = match.group("body")
        cleaned = re.sub(r"\s+", " ", body).strip()
        if not cleaned:
            continue
        chunks.append(IngestChunk(order=idx, text=cleaned, original_tag_type=tag))
    logger.info("Extracted chunks for ingestion", total=len(chunks))
    return chunks


async def _ensure_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    exists = await client.collection_exists(collection_name=collection_name)
    if not exists:
        logger.info("Creating Qdrant collection", collection=collection_name, vector_size=vector_size)
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    info = await client.get_collection(collection_name=collection_name)
    existing_size = _get_vector_size(info)
    if existing_size is not None and existing_size != vector_size:
        raise ValueError(
            f"Existing collection '{collection_name}' vector size {existing_size} does not match required {vector_size}"
        )


def _get_vector_size(info: CollectionInfo) -> Optional[int]:
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        return first.size if first else None
    return vectors.size if vectors else None


def _is_local_qdrant(client: AsyncQdrantClient) -> bool:
    return isinstance(getattr(client, "_client", None), AsyncQdrantLocal)


async def _ensure_text_index(client: AsyncQdrantClient, collection_name: str, field_name: str = "text") -> None:
    if _is_local_qdrant(client):
        logger.warning(
            "Local Qdrant does not support payload indexes; skipping text index creation",
            collection=collection_name,
            field=field_name,
        )
        return
    schema = (await client.get_collection(collection_name=collection_name)).payload_schema or {}
    if field_name in schema:
        return
    params = TextIndexParams(
        type=TextIndexType.TEXT,
        tokenizer=TokenizerType.MULTILINGUAL,
        min_token_len=2,
        lowercase=True,
    )
    logger.info("Creating text index", collection=collection_name, field=field_name)
    await client.create_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        field_schema=params,
    )


async def _embed_chunks(chunks: Sequence[IngestChunk], bundle: EmbeddingsClientBundle) -> Tuple[List[List[float]], int]:
    inputs = [chunk.text for chunk in chunks]
    response = await bundle.client.embeddings.create(model=bundle.model, input=inputs)
    vectors: List[List[float]] = [item.embedding for item in response.data]
    if not vectors or not vectors[0]:
        raise ValueError("Received empty embeddings")
    return vectors, len(vectors[0])


async def _find_similarity(
    client: AsyncQdrantClient, collection_name: str, vector: List[float], threshold: float
) -> Optional[SimilarityMatch]:
    try:
        result = await client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=1,
            score_threshold=threshold,
            with_payload=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Similarity search failed", error=str(exc))
        return None

    if not result.points:
        return None

    point = result.points[0]
    return SimilarityMatch(chunk_id=str(point.id), score=float(point.score))


def _build_points(
    chunks: Sequence[IngestChunk],
    vectors: Sequence[Sequence[float]],
    matches: Sequence[Optional[SimilarityMatch]],
    file_hash: str,
) -> Tuple[List[PointStruct], List[ChunkIngestRecord]]:
    points: List[PointStruct] = []
    records: List[ChunkIngestRecord] = []

    for chunk, vector, match in zip(chunks, vectors, matches, strict=True):
        if match:
            records.append(
                ChunkIngestRecord(
                    chunk_order=chunk.order,
                    chunk_id=match.chunk_id,
                    original_tag_type=chunk.original_tag_type,
                    reused=True,
                    similarity=match.score,
                )
            )
            continue

        chunk_id = str(uuid4())
        payload = {
            "text": chunk.text,
            "source_hash": file_hash,
            "chunk_order": chunk.order,
            "original_tag_type": chunk.original_tag_type,
        }
        points.append(PointStruct(id=chunk_id, vector=list(vector), payload=payload))
        records.append(
            ChunkIngestRecord(
                chunk_order=chunk.order,
                chunk_id=chunk_id,
                original_tag_type=chunk.original_tag_type,
                reused=False,
                similarity=None,
            )
        )

    return points, records


def _init_mapping_store(chunk_mapping_path: Path) -> _SQLiteMappingStore:
    chunk_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{chunk_mapping_path}", future=True)
    metadata = MetaData()
    table = Table(
        "chunk_mappings",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("input_file_hash", String, nullable=False),
        Column("chunk_order", Integer, nullable=False),
        Column("chunk", Text, nullable=False),
        Column("chunk_id", String, nullable=False),
        Column("original_tag_type", String, nullable=False),
        UniqueConstraint("input_file_hash", "chunk_order", name="uq_chunk_order_per_file"),
    )
    metadata.create_all(engine)
    return _SQLiteMappingStore(engine=engine, table=table)


def _upsert_mappings(
    store: _SQLiteMappingStore,
    file_hash: str,
    chunks: Sequence[IngestChunk],
    records: Sequence[ChunkIngestRecord],
) -> None:
    # Use chunk_order to align chunk text with records
    chunk_by_order = {chunk.order: chunk for chunk in chunks}
    with store.engine.begin() as conn:
        for record in records:
            chunk = chunk_by_order[record.chunk_order]
            existing = conn.execute(
                select(store.table.c.id, store.table.c.chunk_id).where(
                    store.table.c.input_file_hash == file_hash,
                    store.table.c.chunk_order == record.chunk_order,
                )
            ).first()
            payload = {
                "chunk": chunk.text,
                "chunk_id": record.chunk_id,
                "original_tag_type": record.original_tag_type,
            }
            if existing:
                conn.execute(
                    update(store.table)
                    .where(store.table.c.id == existing.id)
                    .values(payload)
                )
            else:
                conn.execute(
                    store.table.insert().values(
                        input_file_hash=file_hash,
                        chunk_order=record.chunk_order,
                        **payload,
                    )
                )


async def run_document_post_ingest(
    processed_markdown: str,
    *,
    file_hash: str,
    collection_name: str,
    qdrant_path: Path,
    chunk_mapping_path: Path,
    provider: Optional[str] = None,
    similarity_threshold: float = 0.95,
) -> DocumentIngestResult:
    """
    Ingest processed markdown chunks into a file-based Qdrant collection with SQLite mapping.
    """
    qdrant_path.mkdir(parents=True, exist_ok=True)
    chunks = _extract_chunks(processed_markdown)
    if not chunks:
        logger.warning("No chunks found to ingest", file_hash=file_hash)
        return DocumentIngestResult(
            file_hash=file_hash,
            collection_name=collection_name,
            total_chunks=0,
            inserted_chunks=0,
            reused_chunks=0,
            chunk_mapping_path=chunk_mapping_path,
            chunk_records=[],
        )

    embedding_bundle = get_embeddings_client_bundle(provider=provider)
    qdrant_client = AsyncQdrantClient(path=str(qdrant_path))

    try:
        vectors, vector_size = await _embed_chunks(chunks, embedding_bundle)
        await _ensure_collection(qdrant_client, collection_name, vector_size)
        await _ensure_text_index(qdrant_client, collection_name, field_name="text")

        matches: List[Optional[SimilarityMatch]] = []
        for vector in vectors:
            match = await _find_similarity(qdrant_client, collection_name, vector, similarity_threshold)
            matches.append(match)

        points, records = _build_points(chunks, vectors, matches, file_hash)
        if points:
            upload_method = qdrant_client.upload_points
            if inspect.iscoroutinefunction(upload_method):
                await upload_method(collection_name=collection_name, points=points, wait=True)
            else:
                await asyncio.to_thread(upload_method, collection_name=collection_name, points=points, wait=True)
            logger.info("Uploaded new points", count=len(points))

        store = await asyncio.to_thread(_init_mapping_store, chunk_mapping_path)
        await asyncio.to_thread(_upsert_mappings, store, file_hash, chunks, records)

        inserted = sum(1 for r in records if not r.reused)
        reused = len(records) - inserted
        logger.info(
            "Ingest completed",
            total=len(records),
            inserted=inserted,
            reused=reused,
            collection=collection_name,
        )
        return DocumentIngestResult(
            file_hash=file_hash,
            collection_name=collection_name,
            total_chunks=len(records),
            inserted_chunks=inserted,
            reused_chunks=reused,
            chunk_mapping_path=chunk_mapping_path,
            chunk_records=records,
        )
    finally:
        try:
            await qdrant_client.close()
        except Exception:
            pass
        try:
            await embedding_bundle.client.close()
        except Exception:
            pass


__all__ = ["run_document_post_ingest"]
