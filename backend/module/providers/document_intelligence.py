from __future__ import annotations

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("providers_document_intelligence")


def get_document_intelligence_client() -> DocumentIntelligenceClient:
    logger.info("Creating Azure Document Intelligence client")
    return DocumentIntelligenceClient(
        endpoint=settings.doc_intelligence.endpoint,
        credential=AzureKeyCredential(settings.doc_intelligence.api_key),
    )
