from __future__ import annotations

import asyncio
from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.csv.csv_pipeline_models import CSVConversionConfig, CSVConversionResult
from backend.module.pipelines.csv.csv_pipeline_utils import (
    load_csv_to_arrow,
    resolve_output_path,
    schema_as_dict,
    table_to_dataframe,
    write_parquet_table,
)

logger = LoggingInterceptor("pipelines_csv_pipeline")


async def run_csv_pipeline(config: CSVConversionConfig) -> CSVConversionResult:
    """
    Asynchronously load a CSV into a typed DataFrame + Arrow table.

    Returns the in-memory dataframe plus metadata; writing to disk is optional.
    """
    return await asyncio.to_thread(_run_pipeline_sync, config)


async def run_csv_pipeline_and_save(config: CSVConversionConfig) -> CSVConversionResult:
    """
    Convenience helper: run the pipeline and persist the Parquet output.
    """
    result = await run_csv_pipeline(config)
    write_parquet_table(result.arrow_table, result.output_path)
    return result


def _run_pipeline_sync(config: CSVConversionConfig) -> CSVConversionResult:
    resolved_output = resolve_output_path(config.input_path, config.output_path)
    enriched_config = config.model_copy(update={"output_path": resolved_output})

    logger.info("Starting CSV pipeline", input_path=str(enriched_config.input_path), output_path=str(resolved_output))
    table = load_csv_to_arrow(enriched_config)
    dataframe = table_to_dataframe(table)
    column_types = schema_as_dict(table)

    logger.info(
        "CSV pipeline completed",
        rows=dataframe.shape[0],
        columns=list(dataframe.columns),
        output_path=str(resolved_output),
    )

    return CSVConversionResult(
        dataframe=dataframe,
        arrow_table=table,
        output_path=resolved_output,
        row_count=dataframe.shape[0],
        column_types=column_types,
    )


__all__ = ["run_csv_pipeline", "run_csv_pipeline_and_save"]
