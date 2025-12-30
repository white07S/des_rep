from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.logging import LoggingInterceptor  # noqa: E402
from backend.module.pipelines.csv.csv_pipeline import run_csv_pipeline  # noqa: E402
from backend.module.pipelines.csv.csv_pipeline_models import CSVConversionConfig  # noqa: E402
from backend.module.pipelines.csv.csv_pipeline_utils import write_parquet_table  # noqa: E402

logger = LoggingInterceptor("csv_pipeline_test")

INPUT_FILE_PATH = Path("/Users/preetam/Develop/rag_business/backend/test_files/testing.csv")
OUTPUT_FILE_PATH = Path("/Users/preetam/Develop/rag_business/backend/test_files/testing.parquet")


async def run() -> None:
    config = CSVConversionConfig(input_path=INPUT_FILE_PATH, output_path=OUTPUT_FILE_PATH)
    result = await run_csv_pipeline(config)
    write_parquet_table(result.arrow_table, result.output_path)
    logger.info(
        "CSV pipeline completed and Parquet saved",
        rows=result.row_count,
        output_path=str(result.output_path),
    )
    print(f"[csv_pipeline] saved {result.row_count} rows to {result.output_path}")
    print(f"[csv_pipeline] column types: {result.column_types}")


if __name__ == "__main__":
    asyncio.run(run())
