from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from openai import AsyncAzureOpenAI, AsyncOpenAI
from tqdm import tqdm

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.docs.document_pipeline_models import ExtractedImage
from backend.module.providers.llm import DEFAULT_HEADERS

logger = LoggingInterceptor("pipelines_docs_utils")


def compute_file_hash(file_path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            sha.update(block)
    digest = sha.hexdigest()
    logger.info("Computed file hash", file=str(file_path), hash=digest)
    return digest


def _page_scale_from_azure_unit(azure_page: Dict[str, Any], fitz_page: fitz.Page) -> Tuple[float, float]:
    """
    Returns (scale_x, scale_y) to map Azure polygon coords to PyMuPDF points.
    Azure page 'unit' is commonly 'inch' for PDF, but can vary.
    """
    unit = (azure_page.get("unit") or "").lower()
    az_w = azure_page.get("width")
    az_h = azure_page.get("height")

    if unit in ("inch", "inches"):
        return 72.0, 72.0
    if unit in ("point", "points"):
        return 1.0, 1.0

    if (
        unit in ("pixel", "pixels")
        and isinstance(az_w, (int, float))
        and isinstance(az_h, (int, float))
        and az_w > 0
        and az_h > 0
    ):
        return float(fitz_page.rect.width) / float(az_w), float(fitz_page.rect.height) / float(az_h)

    return 72.0, 72.0


def build_markdown_content(
    analysis: Dict[str, Any],
    pdf_path: Path,
    file_hash: str,
    *,
    image_path_prefix: str = "images",
) -> Tuple[str, List[ExtractedImage]]:
    """
    Build markdown from Azure analysis while extracting figures as in-memory PNGs.

    Returns the markdown string with image links (relative) and a list of ExtractedImage payloads.
    """
    content = analysis.get("content", "")
    if not isinstance(content, str) or not content:
        logger.warning("Analysis content missing or empty; returning empty markdown")
        return "", []

    figures = analysis.get("figures") or []
    pages = analysis.get("pages") or []

    doc = fitz.open(pdf_path)
    replacements: List[Tuple[int, int, str]] = []
    extracted_images: List[ExtractedImage] = []

    images_dir_rel = f"{image_path_prefix}/{file_hash}"

    for idx, fig in enumerate(tqdm(figures, desc="figures", unit="fig")):
        fig_id = fig.get("id") or f"figure-{idx + 1}"
        regions = fig.get("boundingRegions") or []
        if not regions:
            continue

        region = regions[0]
        page_num = (region.get("pageNumber") or 1) - 1
        polygon = region.get("polygon") or []

        if page_num < 0 or page_num >= len(doc):
            logger.warning("Skipping figure with invalid page", figure=fig_id, page=page_num)
            continue
        if len(polygon) < 8:
            logger.warning("Skipping figure with insufficient polygon points", figure=fig_id)
            continue

        page = doc[page_num]
        azure_page = pages[page_num] if page_num < len(pages) else {}
        sx, sy = _page_scale_from_azure_unit(azure_page, page)

        xs = polygon[0::2]
        ys = polygon[1::2]
        min_x, min_y = min(xs), min(ys)
        max_x, max_y = max(xs), max(ys)

        rect = fitz.Rect(min_x * sx, min_y * sy, max_x * sx, max_y * sy)

        try:
            pix = page.get_pixmap(clip=rect)
            image_rel_path = f"{images_dir_rel}/{fig_id}.png"
            image_bytes = pix.tobytes("png")

            extracted_images.append(
                ExtractedImage(
                    figure_id=fig_id,
                    page_number=page_num + 1,
                    suggested_path=image_rel_path,
                    content_bytes=image_bytes,
                )
            )

            replacement_str = f"![{fig_id}]({image_rel_path})"
            for span in fig.get("spans") or []:
                offset = span.get("offset")
                length = span.get("length")
                if isinstance(offset, int) and isinstance(length, int) and length > 0:
                    replacements.append((offset, length, replacement_str))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to extract figure",
                figure=fig_id,
                page=page_num + 1,
                error=str(exc),
            )

    doc.close()

    replacements.sort(key=lambda r: r[0], reverse=True)
    md = content
    base_len = len(content)

    for offset, length, repl in replacements:
        if 0 <= offset <= base_len and 0 <= offset + length <= base_len:
            md = md[:offset] + "\n" + repl + "\n" + md[offset + length :]

    return md, extracted_images


@dataclass
class ResponsesClientBundle:
    client: AsyncOpenAI | AsyncAzureOpenAI
    model: str


def get_responses_client_bundle(
    provider: Optional[str] = None, model_profile: Optional[str] = None
) -> ResponsesClientBundle:
    """
    Build an async Responses-capable client using configured providers.

    Defaults mirror LLM provider selection (openai or azure) and pull model via settings.get_llm_model.
    """
    provider_name = settings.resolve_provider(provider)
    model = settings.get_llm_model(model_profile, provider_name)
    logger.info("Creating responses client", provider=provider_name, model=model)

    if provider_name == "openai":
        client = AsyncOpenAI(
            api_key=settings.openai.api_key,
            base_url=settings.openai.api_base,
            default_headers=DEFAULT_HEADERS,
        )
    else:
        client = AsyncAzureOpenAI(
            api_key=settings.azure.llm.api_key,
            azure_endpoint=settings.azure.api_base,
            api_version=settings.azure.llm.api_version,
            default_headers=DEFAULT_HEADERS,
        )

    return ResponsesClientBundle(client=client, model=model)


def parse_response_json(payload: str) -> Dict[str, Any]:
    """Parse JSON text from a response output, logging helpful diagnostics."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON response", error=str(exc))
        raise


__all__ = [
    "compute_file_hash",
    "build_markdown_content",
    "ResponsesClientBundle",
    "get_responses_client_bundle",
    "parse_response_json",
]
