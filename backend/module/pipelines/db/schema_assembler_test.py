from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.logging import LoggingInterceptor  # noqa: E402
from backend.module.pipelines.db.schema_assembler import assemble_schema  # noqa: E402

logger = LoggingInterceptor("schema_assembler_test")

# Hardcoded inputs (fill these before running)
DB_PATH = Path("/Users/preetam/Develop/rag_business/backend/test_files/department_store.sqlite")
SCHEMA_PATH = Path("/Users/preetam/Develop/rag_business/backend/test_files/schema.sql")
OUTPUT_SPEC_PATH = Path("/Users/preetam/Develop/rag_business/backend/test_files/ai_spec_schema.json")
BUSINESS_CONTEXT = """
This dataset supports end-to-end retail business analysis across the full value chain: it lets you measure customer demand through the order lifecycle (new → pending/partial → completed/cancelled), understand what products and categories drive volume and implied revenue, and segment customers by behavior and payment preferences to spot retention and conversion patterns. On the supply side, it enables procurement and supplier analytics such as supplier concentration risk, sourcing continuity over time, and spend/value allocation across products—useful for cost control and operational resilience. Combined with organizational entities (stores, departments, staff assignments) and shared address history, the data also facilitates operational insights like workforce coverage by department, assignment churn, and geography/footprint perspectives (customer and supplier location dynamics), giving a business view that connects demand, fulfillment outcomes, sourcing decisions, and internal capacity planning.
"""


async def run() -> None:
    schema = await assemble_schema(
        sqlite_path=DB_PATH,
        schema_sql_path=SCHEMA_PATH,
        output_path=OUTPUT_SPEC_PATH,
        business_context=BUSINESS_CONTEXT.strip() or None,
    )
    print(f"[schema] saved to {OUTPUT_SPEC_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
