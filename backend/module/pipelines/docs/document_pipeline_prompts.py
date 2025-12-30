from __future__ import annotations

from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.docs.document_pipeline_models import ProcessingItem

logger = LoggingInterceptor("pipelines_docs_prompts")


def build_image_prompt(item: ProcessingItem) -> str:
    """Construct prompt for image analysis with surrounding context."""
    context = "\n".join(item.context_before + ["[IMAGE HERE]"] + item.context_after)
    logger.info("Building image prompt", identifier=item.identifier, line=item.line_number)
    return (
        "You are analyzing an image from a document. "
        "Use the context to infer what the image represents and capture every visible detail. "
        "Do not mention the file name or path in your description or key elements.\n\n"
        f"Context:\n{context}\n\n"
        f"Image reference: {item.identifier}\n\n"
        "Requirements:\n"
        "- If it is a logo, identify the company/organization.\n"
        "- If the image is a logo or brand mark, set is_logo to true.\n"
        "- Include ALL visible or implied numbers, text, labels, IDs, and specific details.\n"
        "- Do not redact, generalize, or remove any information.\n"
        "- Do not include file names or paths in the description.\n"
        "- Provide a comprehensive description that preserves all important data.\n"
        "- Consider the surrounding text to understand the image's purpose."
    )


def build_table_prompt(item: ProcessingItem) -> str:
    """Construct prompt for table analysis with surrounding context."""
    context = "\n".join(item.context_before + ["[TABLE HERE]"] + item.context_after)
    logger.info("Building table prompt", line=item.line_number)
    return (
        "You are analyzing a table from a document. Use the context to understand the table.\n\n"
        f"Context:\n{context}\n\n"
        f"Table content:\n{item.content}\n\n"
        "Requirements:\n"
        "- Include ALL numbers, percentages, values, and specific data points.\n"
        "- Do not redact, round, or generalize any numerical values.\n"
        "- Preserve column headers and important row labels if present.\n"
        "- Provide a comprehensive summary that retains all critical information.\n"
        "- Identify what this table represents and its purpose in the document."
    )


def truncate_table_content(table_text: str, limit: int = 3000) -> str:
    """Trim overly large table text."""
    if len(table_text) <= limit:
        return table_text
    logger.warning("Truncating large table content", original_length=len(table_text), limit=limit)
    return f"{table_text[:limit]}\n... [truncated]"


__all__ = ["build_image_prompt", "build_table_prompt", "truncate_table_content"]
