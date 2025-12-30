from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.db.schema_assembler_models import TableSpec

logger = LoggingInterceptor("pipelines_db_schema_assembler_utils")

CONTEXTS_DIR = Path(__file__).resolve().parent / "contexts"


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def load_schema_sql(schema_path: Path) -> str:
    logger.info("Loading schema.sql", path=str(schema_path))
    return _read_text(schema_path)


def load_schema_spec_definition(contexts_dir: Path | None = None) -> str:
    ctx_dir = contexts_dir or CONTEXTS_DIR
    path = ctx_dir / "schema_spec.json"
    logger.info("Loading schema_spec.json", path=str(path))
    return _read_text(path)


def load_example_schema(contexts_dir: Path | None = None) -> str:
    ctx_dir = contexts_dir or CONTEXTS_DIR
    path = ctx_dir / "example_schema.json"
    logger.info("Loading example_schema.json", path=str(path))
    return _read_text(path)


def fetch_table_samples(sqlite_path: Path, limit: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {sqlite_path}")

    logger.info("Fetching sample rows from sqlite", path=str(sqlite_path), limit=limit)
    samples: Dict[str, List[Dict[str, Any]]] = {}
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()

        for table_row in table_rows:
            table_name = str(table_row["name"])
            if table_name.startswith("sqlite_"):
                continue
            rows = conn.execute(
                f'SELECT * FROM "{table_name}" LIMIT ?;',  # noqa: S608
                (limit,),
            ).fetchall()
            samples[table_name] = [dict(row) for row in rows]

    return samples


def introspect_sqlite_schema(sqlite_path: Path) -> Dict[str, Dict[str, Any]]:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {sqlite_path}")

    logger.info("Introspecting sqlite schema", path=str(sqlite_path))
    definitions: Dict[str, Dict[str, Any]] = {}
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()

        for table_row in table_rows:
            table_name = str(table_row["name"])
            if table_name.startswith("sqlite_"):
                continue

            columns_info = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            primary_keys = [row["name"] for row in columns_info if row["pk"]]
            columns = [
                {
                    "column_name": row["name"],
                    "column_dtype": row["type"] or "TEXT",
                    "is_primary_key": bool(row["pk"]),
                }
                for row in columns_info
            ]

            fk_rows = conn.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()
            foreign_keys = [
                {
                    "from_table": table_name,
                    "from_column": row["from"],
                    "to_table": row["table"],
                    "to_column": row["to"],
                    "relationship_type": "many-to-one",
                }
                for row in fk_rows
            ]

            definitions[table_name] = {
                "columns": columns,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
            }

    return definitions


def _normalize_pk(pk_value: str | list[str] | None) -> list[str]:
    if isinstance(pk_value, str):
        return [pk_value] if pk_value else []
    if isinstance(pk_value, list):
        return [col for col in pk_value if col]
    return []


def validate_against_sql_definitions(
    tables: List[TableSpec], definitions: Dict[str, Dict[str, Any]]
) -> List[str]:
    issues: List[str] = []
    spec_by_name = {table.table_name: table for table in tables}
    expected_tables = set(definitions.keys())

    missing_tables = expected_tables - set(spec_by_name.keys())
    extra_tables = set(spec_by_name.keys()) - expected_tables
    if missing_tables:
        issues.append(f"Missing tables: {', '.join(sorted(missing_tables))}")
    if extra_tables:
        issues.append(f"Unexpected tables: {', '.join(sorted(extra_tables))}")

    for table_name in expected_tables & set(spec_by_name.keys()):
        spec_table = spec_by_name[table_name]
        definition = definitions[table_name]

        defined_columns = {col["column_name"] for col in definition["columns"]}
        spec_columns = {col.column_name for col in spec_table.columns}
        missing_cols = defined_columns - spec_columns
        extra_cols = spec_columns - defined_columns
        if missing_cols:
            issues.append(f"{table_name}: missing columns {', '.join(sorted(missing_cols))}")
        if extra_cols:
            issues.append(f"{table_name}: unexpected columns {', '.join(sorted(extra_cols))}")

        # Enforce dtype consistency to avoid judge drift.
        def_dtype = {
            col["column_name"]: (col.get("column_dtype") or col.get("type") or "")
            for col in definition["columns"]
        }
        spec_dtype = {col.column_name: col.column_dtype for col in spec_table.columns}
        dtype_mismatches = [
            f"{col} (expected {def_dtype.get(col)}, got {spec_dtype.get(col)})"
            for col in sorted(def_dtype.keys() & spec_dtype.keys())
            if (def_dtype.get(col) or "").upper() != (spec_dtype.get(col) or "").upper()
        ]
        if dtype_mismatches:
            issues.append(f"{table_name}: column_dtype mismatch for {', '.join(dtype_mismatches)}")

        defined_pk = definition["primary_keys"]
        spec_pk = _normalize_pk(spec_table.primary_key)
        if set(defined_pk) != set(spec_pk):
            issues.append(f"{table_name}: primary_key mismatch (expected {defined_pk}, got {spec_pk})")

        defined_fks = {
            (
                fk["from_column"],
                fk["to_table"],
                fk["to_column"],
            )
            for fk in definition["foreign_keys"]
        }
        spec_fks = {(fk.from_column, fk.to_table, fk.to_column) for fk in spec_table.foreign_keys}
        missing_fks = defined_fks - spec_fks
        extra_fks = spec_fks - defined_fks
        if missing_fks:
            issues.append(
                f"{table_name}: missing foreign keys "
                + ", ".join(f"{f[0]}->{f[1]}.{f[2]}" for f in sorted(missing_fks))
            )
        if extra_fks:
            issues.append(
                f"{table_name}: unexpected foreign keys "
                + ", ".join(f"{f[0]}->{f[1]}.{f[2]}" for f in sorted(extra_fks))
            )

    return issues


def current_utc_timestamp() -> str:
    """Return ISO 8601 timestamp with Z suffix."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "current_utc_timestamp",
    "fetch_table_samples",
    "introspect_sqlite_schema",
    "load_example_schema",
    "load_schema_spec_definition",
    "load_schema_sql",
    "validate_against_sql_definitions",
]
