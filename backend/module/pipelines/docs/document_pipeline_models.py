from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("pipelines_docs_models")


class ImageType(str, Enum):
    LOGO = "logo"
    DIAGRAM = "diagram"
    CHART = "chart"
    PHOTO = "photo"
    OTHER = "other"


class ImageAnalysis(BaseModel):
    """Structured output for image analysis."""

    is_logo: bool = Field(description="Whether the image is a logo")
    image_type: ImageType = Field(description="Type of image")
    logo_of: Optional[str] = Field(None, description="If it's a logo, what company/organization")
    description: str = Field(
        description=(
            "Detailed description of the image. Include ALL numbers, IDs, labels, text, and important "
            "details visible in the image. Do not redact or remove any information."
        )
    )
    key_elements: List[str] = Field(
        description="List of key elements, numbers, or identifiers in the image",
        default_factory=list,
    )


class TableAnalysis(BaseModel):
    """Structured output for table analysis."""

    title: Optional[str] = Field(None, description="Title or heading of the table if identifiable")
    purpose: str = Field(description="What this table represents and its purpose")
    key_data: List[str] = Field(
        description="Key data points, numbers, and important values from the table. Include ALL specific numbers and values.",
        default_factory=list,
    )
    summary: str = Field(
        description=(
            "Comprehensive summary of the table content. Include ALL numbers, percentages, values, and "
            "specific data points. Do not redact or generalize any information."
        )
    )
    column_headers: List[str] = Field(
        description="Column headers if available",
        default_factory=list,
    )
    row_count: Optional[int] = Field(None, description="Approximate number of rows")


class ExtractedImage(BaseModel):
    """Container for extracted figure images."""

    figure_id: str
    page_number: int
    suggested_path: str
    content_bytes: bytes


class DocumentPreprocessResult(BaseModel):
    """Outputs from the pre-pipeline."""

    file_hash: str
    original_filename: str
    analysis: Dict[str, Any]
    markdown: str
    images: List[ExtractedImage] = Field(default_factory=list)


class ProcessingItem(BaseModel):
    """Normalized processing item for post-processing."""

    item_type: str  # 'image' | 'table'
    identifier: str
    content: str
    context_before: List[str]
    context_after: List[str]
    start_pos: int
    end_pos: int
    original_match: str
    line_number: int


class ProcessedReplacement(BaseModel):
    item: ProcessingItem
    replacement_text: str


class DocumentPostprocessResult(BaseModel):
    """Outputs from the post-pipeline."""

    processed_markdown: str
    replacements: List[ProcessedReplacement] = Field(default_factory=list)
    total_items: int = 0
    images_count: int = 0
    tables_count: int = 0


class ChunkIngestRecord(BaseModel):
    """Metadata captured for each ingested (or reused) chunk."""

    chunk_order: int
    chunk_id: str
    original_tag_type: str
    reused: bool = Field(
        description="True when an existing chunk was reused due to similarity threshold."
    )
    similarity: Optional[float] = Field(
        default=None, description="Similarity score when a chunk was reused."
    )


class DocumentIngestResult(BaseModel):
    """Outputs from the ingest stage."""

    file_hash: str
    collection_name: str
    total_chunks: int
    inserted_chunks: int
    reused_chunks: int
    chunk_mapping_path: Path
    chunk_records: List[ChunkIngestRecord] = Field(default_factory=list)


class DocumentPipelineResult(BaseModel):
    """Combined outputs from pre and post stages."""

    pre: DocumentPreprocessResult
    post: DocumentPostprocessResult
    ingest: DocumentIngestResult


__all__ = [
    "ImageType",
    "ImageAnalysis",
    "TableAnalysis",
    "ExtractedImage",
    "DocumentPreprocessResult",
    "ProcessingItem",
    "ProcessedReplacement",
    "DocumentPostprocessResult",
    "ChunkIngestRecord",
    "DocumentIngestResult",
    "DocumentPipelineResult",
]
