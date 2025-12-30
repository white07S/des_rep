from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor
from backend.module.providers.embeddings import get_embeddings_client_bundle

logger = LoggingInterceptor("mcp_docs_tools")


def _artifacts_root() -> Path:
    base = settings.storage.data_root / "artifacts"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _meta_files_path() -> Path:
    return settings.storage.data_root / "metadata" / "files.json"


def list_docs(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List document file_ids (PDF uploads) with user_id.
    """
    meta_path = _meta_files_path()
    if not meta_path.exists():
        return []
    with meta_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    docs: List[Dict[str, Any]] = []
    for file_id, record in payload.items():
        if record.get("file_type") != "pdf":
            continue
        user = record.get("user")
        if user_id and user != user_id:
            continue
        docs.append({"file_id": file_id, "user_id": user, "original_filename": record.get("original_filename")})
    logger.info("Listed docs", count=len(docs), user_filter=user_id or "all")
    return sorted(docs, key=lambda x: (x["user_id"] or "", x["file_id"]))


def _locate_doc_paths(file_id: str, user_id: Optional[str] = None) -> Tuple[Path, Path, str]:
    root = _artifacts_root()
    matches: List[Tuple[Path, Path, str]] = []
    if user_id:
        base = root / user_id / file_id
        qdrant_path = base / "qdrant"
        mapping_path = base / "chunk_mapping.sqlite"
        if qdrant_path.exists() and mapping_path.exists():
            return qdrant_path.resolve(), mapping_path.resolve(), user_id
    for mapping_path in root.glob(f"*/{file_id}/chunk_mapping.sqlite"):
        qdrant_path = mapping_path.parent / "qdrant"
        if qdrant_path.exists():
            matches.append((qdrant_path.resolve(), mapping_path.resolve(), mapping_path.parent.parent.name))
    if not matches:
        raise FileNotFoundError(f"No artifacts found for file_id={file_id}")
    if len(matches) > 1:
        raise ValueError(f"Multiple artifacts found for file_id={file_id}; specify user_id.")
    return matches[0]


def _resolve_collection_name(qdrant_path: Path) -> str:
    meta_path = qdrant_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Qdrant meta.json not found at {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    collections = data.get("collections") or {}
    if not collections:
        raise ValueError(f"No collections defined in {meta_path}")
    return next(iter(collections.keys()))


async def _embed_query(text: str, provider: Optional[str] = None) -> List[float]:
    bundle = get_embeddings_client_bundle(provider=provider)
    try:
        response = await bundle.client.embeddings.create(model=bundle.model, input=[text])
        vector = response.data[0].embedding
        if not vector:
            raise ValueError("Empty embedding returned for query")
        return list(vector)
    finally:
        try:
            await bundle.client.close()
        except Exception:
            pass


def _connect_mapping_db(mapping_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{mapping_path}", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _fts_score(chunks: List[Dict[str, Any]], query: str) -> Dict[str, float]:
    tokens = {t.lower() for t in query.split() if t.strip()}
    scores: Dict[str, float] = {}
    if not tokens:
        return scores
    for chunk in chunks:
        text = str(chunk.get("chunk") or "").lower()
        count = sum(text.count(tok) for tok in tokens)
        if count:
            scores[chunk["chunk_id"]] = float(count)
    return scores


def _fetch_chunks(mapping_path: Path) -> List[Dict[str, Any]]:
    with _connect_mapping_db(mapping_path) as conn:
        rows = conn.execute(
            "SELECT chunk_id, chunk_order, chunk, original_tag_type FROM chunk_mappings ORDER BY chunk_order ASC"
        ).fetchall()
        return [dict(row) for row in rows]


def _neighbors(
    chunks: List[Dict[str, Any]],
    center_order: int,
    window: int = 2,
) -> List[Dict[str, Any]]:
    start = max(0, center_order - window)
    end = min(len(chunks) - 1, center_order + window)
    slice_rows = [c for c in chunks if start <= c["chunk_order"] <= end]
    return slice_rows


def _merge_scores(
    embed_points: List[ScoredPoint],
    text_scores: Dict[str, float],
    chunks_by_id: Dict[str, Dict[str, Any]],
    k: int = 5,
) -> List[Tuple[str, float]]:
    combined: Dict[str, float] = {}
    for point in embed_points:
        cid = str(point.id)
        combined[cid] = float(point.score)
    for cid, score in text_scores.items():
        combined[cid] = combined.get(cid, 0.0) + score
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    filtered = [(cid, score) for cid, score in ranked if cid in chunks_by_id]
    return filtered[:k]


async def search_docs(
    query: str,
    *,
    file_id: str,
    user_id: Optional[str] = None,
    provider: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Hybrid search (embeddings + simple text match) returning top contexts.
    """
    qdrant_path, mapping_path, owner = _locate_doc_paths(file_id, user_id)
    try:
        collection_name = _resolve_collection_name(qdrant_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falling back to text-only search; collection unavailable", file_id=file_id, error=str(exc))
        collection_name = None
    chunks = _fetch_chunks(mapping_path)
    if not chunks:
        return []
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    # Embedding search
    embed_points: List[ScoredPoint] = []
    if collection_name:
        try:
            vector = await _embed_query(query, provider=provider)
            client = QdrantClient(path=str(qdrant_path))
            try:
                embed_points = client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=True,
                )
            finally:
                try:
                    client.close()
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding search failed; falling back to text only", error=str(exc))
            embed_points = []

    # Text match scores
    text_scores = _fts_score(chunks, query)

    merged = _merge_scores(embed_points, text_scores, chunks_by_id, k=top_k)

    results: List[Dict[str, Any]] = []
    for cid, score in merged:
        center = chunks_by_id[cid]
        context = _neighbors(chunks, center["chunk_order"], window=2)
        results.append(
            {
                "chunk_id": cid,
                "score": score,
                "file_id": file_id,
                "user_id": owner,
                "chunk_order": center["chunk_order"],
                "context": [
                    {
                        "chunk_id": c["chunk_id"],
                        "chunk_order": c["chunk_order"],
                        "text": c["chunk"],
                        "original_tag_type": c["original_tag_type"],
                    }
                    for c in context
                ],
            }
        )
    logger.info("Search completed", file_id=file_id, user_id=owner, results=len(results))
    return results


def _discover_sample_doc_entry() -> Tuple[str, str] | None:
    docs = list_docs()
    if not docs:
        return None
    doc = docs[0]
    return doc["user_id"], doc["file_id"]


def _run_smoke_tests() -> None:
    sample = _discover_sample_doc_entry()
    if not sample:
        print("No docs found in metadata; skipping smoke tests.")
        return
    user_id, file_id = sample
    print(f"[test] doc sample -> user={user_id} file_id={file_id}")

    try:
        docs = list_docs(user_id)
        print(f"[test] list_docs -> {docs}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] list_docs error: {exc}")
        return

    try:
        results = asyncio.run(search_docs("Average deposit balances ", file_id=file_id, user_id=user_id))
        print(f"Results: {results}d ")
        print(f"[test] search_docs results={len(results)}")
        if results:
            print(f"[test] first result chunk_order={results[0]['chunk_order']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] search_docs error: {exc}")


if __name__ == "__main__":
    _run_smoke_tests()
