from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.csv.csv_pipeline_models import CSVConversionConfig, CSVPipelineError, CSVReadError, CSVWriteError

logger = LoggingInterceptor("pipelines_csv_utils")


def validate_input_path(path: Path) -> Path:
    """Ensure the input path exists, is a file, and points to a CSV."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise CSVReadError(f"Input CSV not found at {resolved}")
    if not resolved.is_file():
        raise CSVReadError(f"Input path is not a file: {resolved}")
    if resolved.suffix.lower() != ".csv":
        raise CSVReadError(f"Input path must be a .csv file: {resolved}")
    return resolved


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    """Return the desired Parquet output path (defaults to input stem with .parquet)."""
    target = output_path or input_path.with_suffix(".parquet")
    return target.expanduser().resolve()


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _derive_cast_overrides(
    columns: Sequence[str], dtype_overrides: Mapping[str, str], auto_cast_dates: bool
) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if auto_cast_dates:
        for col in columns:
            lower = col.lower()
            if "date" in lower:
                overrides[col] = "DATE"
            elif "timestamp" in lower or lower.endswith("_at") or lower.endswith("_time"):
                overrides[col] = "TIMESTAMP"
    overrides.update(dtype_overrides)
    return overrides


def _apply_casts(relation: duckdb.DuckDBPyRelation, cast_overrides: Mapping[str, str]) -> duckdb.DuckDBPyRelation:
    expressions = []
    for col in relation.columns:
        identifier = _quote_identifier(col)
        dtype = cast_overrides.get(col)
        if dtype:
            expressions.append(f"CAST({identifier} AS {dtype}) AS {identifier}")
        else:
            expressions.append(identifier)
    projected = relation.select(", ".join(expressions))
    return projected


def load_csv_to_arrow(config: CSVConversionConfig) -> pa.Table:
    """
    Load a CSV into an Arrow table via DuckDB with optional dtype overrides.
    Raises CSVReadError on failure.
    """
    input_path = validate_input_path(config.input_path)
    logger.info(
        "Loading CSV via DuckDB",
        input_path=str(input_path),
        sample_size=config.sample_size,
        auto_cast_dates=config.auto_cast_dates,
        overrides=bool(config.dtype_overrides),
    )

    connection = duckdb.connect(database=":memory:")
    try:
        relation = connection.from_csv_auto(str(input_path), sample_size=config.sample_size)
        cast_overrides = _derive_cast_overrides(relation.columns, config.dtype_overrides, config.auto_cast_dates)
        if cast_overrides:
            relation = _apply_casts(relation, cast_overrides)

        table = relation.to_arrow_table()
        logger.info(
            "CSV loaded",
            rows=table.num_rows,
            columns=list(table.column_names),
            schema=str(table.schema),
        )
        return table
    except duckdb.Error as exc:
        logger.exception("DuckDB failed to parse CSV", error=str(exc))
        raise CSVReadError(f"DuckDB failed to parse CSV: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error while loading CSV", error=str(exc))
        raise CSVReadError(f"Unexpected error while loading CSV: {exc}") from exc
    finally:
        connection.close()


def table_to_dataframe(table: pa.Table) -> pd.DataFrame:
    """Convert Arrow table to pandas DataFrame while preserving logical dtypes."""
    try:
        return table.to_pandas(date_as_object=False, types_mapper=None)
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
        logger.exception("Failed to convert Arrow table to DataFrame", error=str(exc))
        raise CSVPipelineError(f"Failed to convert Arrow table to DataFrame: {exc}") from exc


def write_parquet_table(table: pa.Table, output_path: Path) -> Path:
    """Persist an Arrow table to Parquet."""
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    try:
        pq.write_table(table, resolved, compression="snappy")
        logger.info("Parquet file written", output_path=str(resolved), rows=table.num_rows)
        return resolved
    except (pa.ArrowInvalid, pa.ArrowIOError, OSError) as exc:
        logger.exception("Failed to write Parquet", error=str(exc))
        raise CSVWriteError(f"Failed to write Parquet file {resolved}: {exc}") from exc


def schema_as_dict(table: pa.Table) -> Dict[str, str]:
    return {field.name: str(field.type) for field in table.schema}


__all__ = [
    "load_csv_to_arrow",
    "resolve_output_path",
    "schema_as_dict",
    "table_to_dataframe",
    "validate_input_path",
    "write_parquet_table",
]
