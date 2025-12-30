from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Iterable, List, Optional, Type, TypeVar

from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.docs.document_pipeline_models import (
    DocumentPostprocessResult,
    ImageAnalysis,
    ProcessingItem,
    ProcessedReplacement,
    TableAnalysis,
)
from backend.module.pipelines.docs.document_pipeline_prompts import (
    build_image_prompt,
    build_table_prompt,
    truncate_table_content,
)
from backend.module.pipelines.docs.document_pipeline_utils import get_responses_client_bundle

logger = LoggingInterceptor("pipelines_docs_post")

CONTEXT_LINES = 5
MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 1000
REQUEST_TIMEOUT = 30.0
TModel = TypeVar("TModel", bound=BaseModel)
MIN_CHUNK_WORDS = 50
MAX_CHUNK_WORDS = 170


def _remove_all_html_comments(content: str) -> str:
    """Strip HTML comments and tidy whitespace."""
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    content = re.sub(r"<p>\s*</p>", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)
    return content


def _find_processable_items(content: str) -> List[ProcessingItem]:
    """Find image and table blocks in order."""
    items: List[ProcessingItem] = []
    lines = content.split("\n")

    image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    markdown_table_pattern = r"(\|[^\n]+\|\n(?:\|[-:\s]+\|\n)?(?:\|[^\n]+\|\n)+)"
    html_table_pattern = r"<table[^>]*>.*?</table>"

    for match in re.finditer(image_pattern, content):
        line_num = content[: match.start()].count("\n")
        context_before = lines[max(0, line_num - CONTEXT_LINES) : line_num]
        context_after = lines[line_num + 1 : min(len(lines), line_num + 1 + CONTEXT_LINES)]
        image_path = match.group(2)
        image_name = Path(image_path).name if image_path else "unknown_image"
        items.append(
            ProcessingItem(
                item_type="image",
                identifier=image_name,
                content=image_name,
                context_before=context_before,
                context_after=context_after,
                start_pos=match.start(),
                end_pos=match.end(),
                original_match=match.group(0),
                line_number=line_num,
            )
        )

    for match in re.finditer(markdown_table_pattern, content, re.MULTILINE):
        line_num = content[: match.start()].count("\n")
        table_lines = match.group(0).count("\n")
        context_before = lines[max(0, line_num - CONTEXT_LINES) : line_num]
        context_after = lines[
            line_num + table_lines + 1 : min(len(lines), line_num + table_lines + 1 + CONTEXT_LINES)
        ]
        items.append(
            ProcessingItem(
                item_type="table",
                identifier=f"table-{len(items)+1}",
                content=match.group(0),
                context_before=context_before,
                context_after=context_after,
                start_pos=match.start(),
                end_pos=match.end(),
                original_match=match.group(0),
                line_number=line_num,
            )
        )

    for match in re.finditer(html_table_pattern, content, re.DOTALL | re.IGNORECASE):
        line_num = content[: match.start()].count("\n")
        table_lines = match.group(0).count("\n")
        context_before = lines[max(0, line_num - CONTEXT_LINES) : line_num]
        context_after = lines[
            line_num + table_lines + 1 : min(len(lines), line_num + table_lines + 1 + CONTEXT_LINES)
        ]
        table_html = match.group(0)
        table_text = re.sub(r"<th[^>]*>", " [Header] ", table_html)
        table_text = re.sub(r"<td[^>]*>", " [Cell] ", table_text)
        table_text = re.sub(r"</t[hd]>", " | ", table_text)
        table_text = re.sub(r"</tr>", "\n", table_text)
        table_text = re.sub(r"<[^>]+>", "", table_text)
        items.append(
            ProcessingItem(
                item_type="table",
                identifier=f"table-html-{len(items)+1}",
                content=table_text,
                context_before=context_before,
                context_after=context_after,
                start_pos=match.start(),
                end_pos=match.end(),
                original_match=match.group(0),
                line_number=line_num,
            )
        )

    items.sort(key=lambda x: x.start_pos)
    logger.info(
        "Found processable items",
        total=len(items),
        images=sum(1 for i in items if i.item_type == "image"),
        tables=sum(1 for i in items if i.item_type == "table"),
    )
    return items


async def _call_structured_response(
    prompt: str,
    *,
    schema_model: Type[TModel],
    client,
    model: str,
    max_output_tokens: int = 500,
) -> TModel:
    response = await asyncio.wait_for(
        client.responses.parse(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            temperature=0.1,
            max_output_tokens=max_output_tokens,
            text_format=schema_model,
        ),
        timeout=REQUEST_TIMEOUT,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("Empty parsed response")
    return parsed


async def _process_image_item(
    item: ProcessingItem,
    client,
    model: str,
) -> ProcessedReplacement:
    prompt = build_image_prompt(item)
    try:
        analysis = await _call_structured_response(
            prompt,
            schema_model=ImageAnalysis,
            client=client,
            model=model,
            max_output_tokens=400,
        )
        if analysis.is_logo:
            replacement = f"<pai>[Logo: {analysis.logo_of}] {analysis.description}</pai>"
        else:
            replacement = f"<pai>{analysis.description}"
            if analysis.key_elements:
                replacement += f" Key elements: {', '.join(analysis.key_elements)}"
            replacement += "</pai>"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image processing failed", error=str(exc), identifier=item.identifier)
        replacement = f"<pai>[Image: {item.identifier}]</pai>"
    return ProcessedReplacement(item=item, replacement_text=replacement)


async def _process_table_item(
    item: ProcessingItem,
    client,
    model: str,
) -> ProcessedReplacement:
    truncated_content = truncate_table_content(item.content)
    prompt_item = item.model_copy(update={"content": truncated_content})
    prompt = build_table_prompt(prompt_item)
    try:
        analysis = await _call_structured_response(
            prompt,
            schema_model=TableAnalysis,
            client=client,
            model=model,
            max_output_tokens=600,
        )
        replacement = f"<pai>[Table"
        if analysis.title:
            replacement += f": {analysis.title}"
        replacement += f"] {analysis.purpose} {analysis.summary}"
        if analysis.key_data:
            replacement += f" Key data points: {', '.join(analysis.key_data)}"
        replacement += "</pai>"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Table processing failed", error=str(exc), identifier=item.identifier)
        replacement = _build_table_fallback(item)
    return ProcessedReplacement(item=item, replacement_text=replacement)


def _apply_replacements(content: str, replacements: List[ProcessedReplacement]) -> str:
    replacements_sorted = sorted(replacements, key=lambda x: x.item.start_pos, reverse=True)
    updated = content
    for repl in replacements_sorted:
        updated = updated[: repl.item.start_pos] + repl.replacement_text + updated[repl.item.end_pos :]
    return updated


def _build_table_fallback(item: ProcessingItem) -> str:
    """Provide a concise fallback description when table parsing fails."""
    def _clean(text: str, limit_words: int) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        return " ".join(words[:limit_words])

    context_window = _clean(" ".join(item.context_before + item.context_after), 40)
    table_preview = _clean(item.content, 40)
    parts: List[str] = ["<pai>[Table: content preserved]"]
    if context_window:
        parts.append(context_window)
    if table_preview:
        parts.append(table_preview)
    parts.append("</pai>")
    return " ".join(parts)


def _dedupe_logo_blocks(lines: Iterable[str]) -> List[str]:
    """Remove duplicate logo <pai> blocks by normalized logo tag."""
    seen: set[str] = set()
    deduped: List[str] = []
    logo_prefix = re.compile(r"<pai>\s*\[logo:\s*([^\]]+)\]", re.IGNORECASE)
    for line in lines:
        match = logo_prefix.match(line.strip())
        if match:
            key = match.group(1).strip().lower()
            if key in seen:
                continue
            seen.add(key)
        deduped.append(line)
    return deduped


def _normalize_paragraphs(markdown: str) -> List[str]:
    """
    Normalize HTML-ish paragraphs and preserve structural hints for chunking.
    - Strip <p>...</p> while keeping content.
    - Keep headings/list markers as separate lines.
    - Merge dangling short lines when they clearly belong together.
    """
    cleaned: List[str] = []
    lines = markdown.split("\n")
    p_tag = re.compile(r"^<p>(.*)</p>$", re.IGNORECASE)

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if "<pai>" in line:
            cleaned.append(line)
            continue

        m = p_tag.match(line)
        if m:
            line = m.group(1).strip()
        # If the line is a heading or starts with bullet/number, keep as-is
        if line.startswith("#"):
            cleaned.append(line)
            continue
        if re.match(r"^[-•·\*]\s+", line) or re.match(r"^\d+\.\s+", line):
            cleaned.append(line)
            continue

        cleaned.append(line)

    # Merge continuation lines where previous ended with ":" or looks like intro
    merged: List[str] = []
    for line in cleaned:
        if merged:
            prev = merged[-1]
            if (
                prev.endswith(":")
                or re.match(r".*(Highlights|Overview|Summary)\s*$", prev, re.IGNORECASE)
            ) and not line.startswith(("<pai>", "#", "-", "•", "·", "*", "1.", "2.", "3.")):
                merged[-1] = f"{prev} {line}"
                continue
        merged.append(line)

    # Merge split sentences/fragments when neither side is structural
    glued: List[str] = []
    for line in merged:
        if (
            glued
            and not line.startswith(("<pai>", "#", "-", "•", "·", "*", "1.", "2.", "3."))
            and not glued[-1].startswith(("<pai>", "#", "-", "•", "·", "*", "1.", "2.", "3."))
            and not glued[-1].rstrip().endswith((".", "!", "?", ";"))
        ):
            glued[-1] = f"{glued[-1]} {line}"
        else:
            glued.append(line)

    # Merge very short fragments into previous line when safe
    fused: List[str] = []
    for line in glued:
        if (
            fused
            and len(line) < 40
            and not line.startswith(("<pai>", "#", "-", "•", "·", "*", "1.", "2.", "3."))
            and not fused[-1].startswith("<pai>")
        ):
            fused[-1] = f"{fused[-1]} {line}"
            continue
        fused.append(line)
    return fused


def _create_meaningful_chunks(content: str) -> str:
    lines = _normalize_paragraphs(content)
    lines = _dedupe_logo_blocks(lines)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_size = 0
    bullet_mode = False

    def _count_words(text: str) -> int:
        return len([w for w in re.split(r"\s+", text.strip()) if w])

    def flush_chunk(force: bool = False) -> None:
        nonlocal current_chunk, current_size, bullet_mode
        if current_chunk:
            chunk_lines: List[str] = []
            for line in current_chunk:
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r"^#{1,6}\s+", "", line)
                line = re.sub(r"^[•·\-\*\+]\s+", "", line)
                line = re.sub(r"^\d+\.\s+", "", line)
                chunk_lines.append(line)
            if chunk_lines:
                chunk_text = " ".join(chunk_lines)
                chunk_text = re.sub(r"\s+", " ", chunk_text)
                words = _count_words(chunk_text)
                if not force and words < MIN_CHUNK_WORDS:
                    return
                if force and words < MIN_CHUNK_WORDS and chunks and chunks[-1].startswith("<p>"):
                    prev = chunks.pop()
                    prev_text = prev[3:-4]
                    combined = f"{prev_text} {chunk_text}"
                    chunks.append(f"<p>{combined}</p>")
                else:
                    chunks.append(f"<p>{chunk_text}</p>")
            current_chunk = []
            current_size = 0
            bullet_mode = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if "<pai>" in line:
            flush_chunk()
            chunks.append(line.strip())
            i += 1
            continue
        if stripped.startswith("<pai>") and stripped.endswith("</pai>"):
            flush_chunk(force=True)
            chunks.append(stripped)
            i += 1
            continue

        # Headings trigger flush and standalone chunk
        if stripped.startswith("#"):
            flush_chunk(force=False)
            heading_text = stripped.lstrip("#").strip()
            current_chunk.append(heading_text)
            current_size += len(heading_text)
            i += 1
            continue

        is_bullet = bool(re.match(r"^[-•·\*]\s+", stripped)) or bool(re.match(r"^\d+\.\s+", stripped))

        # Separate transitions between bullet and non-bullet to preserve layout
        if current_chunk:
            if is_bullet and not bullet_mode:
                flush_chunk(force=True)
            if bullet_mode and not is_bullet:
                flush_chunk(force=True)

        if is_bullet:
            bullet_mode = True

        line_size = len(stripped)
        if current_size > 0:
            if current_size + line_size > MAX_CHUNK_SIZE:
                if current_size >= MIN_CHUNK_SIZE:
                    flush_chunk(force=True)
                    current_chunk.append(stripped)
                    current_size = line_size
                else:
                    current_chunk.append(stripped)
                    current_size += line_size
            else:
                current_chunk.append(stripped)
                current_size += line_size
                if (
                    stripped.endswith((".", "!", "?"))
                    and _count_words(" ".join(current_chunk)) >= MIN_CHUNK_WORDS
                    and i + 1 < len(lines)
                ):
                    next_line = lines[i + 1].strip()
                    if next_line and (next_line[0].isupper() or next_line.startswith("#") or next_line.startswith(("-", "•", "·", "*"))):
                        flush_chunk(force=False)
        else:
            current_chunk.append(stripped)
            current_size = line_size
        i += 1
    flush_chunk(force=True)
    return "\n\n".join(chunks)


async def run_document_post_pipeline(
    markdown: str,
    *,
    provider: Optional[str] = None,
    model_profile: Optional[str] = None,
    max_concurrency: int = 3,
) -> DocumentPostprocessResult:
    """Process markdown to normalized <p>/<pai> output using OpenAI Responses."""
    cleaned = _remove_all_html_comments(markdown)
    items = _find_processable_items(cleaned)
    bundle = get_responses_client_bundle(provider=provider, model_profile=model_profile)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def runner(item: ProcessingItem) -> ProcessedReplacement:
        async with semaphore:
            if item.item_type == "image":
                return await _process_image_item(item, bundle.client, bundle.model)
            return await _process_table_item(item, bundle.client, bundle.model)

    replacements: List[ProcessedReplacement] = []
    if items:
        tasks = [runner(item) for item in items]
        results = await tqdm_asyncio.gather(
            *tasks,
            total=len(tasks),
            desc="processing items",
            unit="item",
        )
        replacements = [r for r in results if isinstance(r, ProcessedReplacement)]

    updated_markdown = _apply_replacements(cleaned, replacements) if replacements else cleaned
    chunked = _create_meaningful_chunks(updated_markdown)

    try:
        await bundle.client.close()
    except Exception:
        pass

    images_count = sum(1 for r in replacements if r.item.item_type == "image")
    tables_count = sum(1 for r in replacements if r.item.item_type == "table")
    logger.info(
        "Post-pipeline finished",
        total_items=len(items),
        replacements=len(replacements),
        images=images_count,
        tables=tables_count,
    )
    return DocumentPostprocessResult(
        processed_markdown=chunked,
        replacements=replacements,
        total_items=len(items),
        images_count=images_count,
        tables_count=tables_count,
    )


__all__ = ["run_document_post_pipeline"]
