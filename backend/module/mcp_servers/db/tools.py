from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("mcp_db_tools")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _artifacts_root() -> Path:
    base = settings.storage.data_root / "artifacts"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _validate_identifier(value: str, kind: str) -> None:
    if not _IDENTIFIER.match(value):
        raise ValueError(f"Invalid {kind} name '{value}'. Use bare identifiers only.")


def _list_sqlite_dirs(user_id: Optional[str] = None) -> List[Path]:
    root = _artifacts_root()
    targets: List[Path] = []
    if user_id:
        candidates = [root / user_id]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir()]
    for user_dir in candidates:
        if not user_dir.exists():
            continue
        for maybe_db in user_dir.iterdir():
            if maybe_db.is_dir() and (maybe_db / "source.sqlite").exists():
                targets.append(maybe_db)
    return targets


def list_databases(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available SQLite artifacts (file_id + user_id).
    """
    dirs = _list_sqlite_dirs(user_id)
    results = [{"file_id": p.name, "user_id": p.parent.name} for p in dirs]
    logger.info("Listed sqlite artifacts", count=len(results), user_filter=user_id or "all")
    return sorted(results, key=lambda x: (x["user_id"], x["file_id"]))


def _locate_sqlite(file_id: str, user_id: Optional[str] = None) -> Tuple[Path, str]:
    root = _artifacts_root()
    matches: List[Tuple[Path, str]] = []
    if user_id:
        candidate = root / user_id / file_id / "source.sqlite"
        if candidate.exists():
            return candidate.resolve(), user_id
    for path in root.glob(f"*/{file_id}/source.sqlite"):
        matches.append((path.resolve(), path.parent.parent.name))
    if not matches:
        raise FileNotFoundError(f"No sqlite artifact found for file_id={file_id}")
    if len(matches) > 1:
        raise ValueError(f"Multiple sqlite artifacts found for file_id={file_id}; please specify user_id")
    return matches[0]


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(file_id: str, user_id: Optional[str] = None) -> List[str]:
    sqlite_path, owner = _locate_sqlite(file_id, user_id)
    with _connect(sqlite_path) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row["name"] for row in cursor.fetchall() if not str(row["name"]).startswith("sqlite_")]
    logger.info("Listed tables", file_id=file_id, user_id=owner, count=len(tables))
    return tables


def sample_data_table(
    file_id: str,
    table_name: str,
    *,
    user_id: Optional[str] = None,
    position: str = "head",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    position_norm = position.lower()
    if position_norm not in {"head", "tail"}:
        raise ValueError("position must be 'head' or 'tail'")
    _validate_identifier(table_name, "table")

    sqlite_path, owner = _locate_sqlite(file_id, user_id)
    with _connect(sqlite_path) as conn:
        order_clause = "" if position_norm == "head" else "ORDER BY ROWID DESC"
        sql = f'SELECT * FROM "{table_name}" {order_clause} LIMIT ?'
        cursor = conn.execute(sql, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        if position_norm == "tail":
            rows = list(reversed(rows))
    logger.info("Sampled table", file_id=file_id, table=table_name, user_id=owner, rows=len(rows), mode=position_norm)
    return rows


def get_column_stats(
    file_id: str,
    table_name: str,
    column_name: str,
    *,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    _validate_identifier(table_name, "table")
    _validate_identifier(column_name, "column")

    sqlite_path, owner = _locate_sqlite(file_id, user_id)
    with _connect(sqlite_path) as conn:
        table_check = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
        ).fetchone()
        if not table_check:
            raise ValueError(f"Table '{table_name}' not found in db {file_id}")

        columns = [row["name"] for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()]
        if column_name not in columns:
            raise ValueError(f"Column '{column_name}' not found in table '{table_name}'")

        def scalar(query: str, params: tuple[Any, ...] = ()) -> int:
            row = conn.execute(query, params).fetchone()
            return int(row[0]) if row and row[0] is not None else 0

        total_rows = scalar(f'SELECT COUNT(*) FROM "{table_name}"')
        null_count = scalar(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{column_name}" IS NULL')
        empty_count = scalar(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{column_name}" = ""')
        nan_count = scalar(
            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{column_name}" != "{column_name}"'
        )
        distinct_count = scalar(f'SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}"')

        top_values: List[Dict[str, Any]] = []
        if distinct_count:
            limit = min(10, max(1, math.ceil(distinct_count * 0.2)))
            top_sql = (
                f'SELECT "{column_name}" AS value, COUNT(*) AS frequency '
                f'FROM "{table_name}" '
                f'WHERE "{column_name}" IS NOT NULL '
                f'GROUP BY "{column_name}" '
                f'ORDER BY frequency DESC '
                f"LIMIT ?"
            )
            for value, freq in conn.execute(top_sql, (limit,)).fetchall():
                freq_int = int(freq)
                pct = (freq_int / total_rows) * 100 if total_rows else 0.0
                try:
                    json.dumps(value)
                    safe_value = value
                except TypeError:
                    safe_value = str(value)
                top_values.append({"value": safe_value, "frequency": freq_int, "frequency_pct": pct})

        stats = {
            "file_id": file_id,
            "user_id": owner,
            "table": table_name,
            "column": column_name,
            "row_count": total_rows,
            "null_count": null_count,
            "null_pct": (null_count / total_rows) * 100 if total_rows else 0.0,
            "empty_string_count": empty_count,
            "empty_string_pct": (empty_count / total_rows) * 100 if total_rows else 0.0,
            "nan_count": nan_count,
            "nan_pct": (nan_count / total_rows) * 100 if total_rows else 0.0,
            "distinct_count": distinct_count,
            "top_values": top_values,
        }
    logger.info("Computed column stats", file_id=file_id, table=table_name, column=column_name, user_id=owner)
    return stats


def get_table_details(
    file_id: str,
    table_name: str,
    *,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    _validate_identifier(table_name, "table")
    sqlite_path, owner = _locate_sqlite(file_id, user_id)

    with _connect(sqlite_path) as conn:
        columns = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        if not columns:
            raise ValueError(f"Table '{table_name}' not found in db {file_id}")
        fks = conn.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()

        column_defs = [
            {
                "name": col["name"],
                "type": col["type"],
                "notnull": bool(col["notnull"]),
                "default": col["dflt_value"],
                "primary_key": bool(col["pk"]),
            }
            for col in columns
        ]
        foreign_keys = [
            {
                "from_column": fk["from"],
                "to_table": fk["table"],
                "to_column": fk["to"],
                "on_update": fk["on_update"],
                "on_delete": fk["on_delete"],
            }
            for fk in fks
        ]

    logger.info("Fetched table details", file_id=file_id, table=table_name, user_id=owner)
    return {
        "file_id": file_id,
        "user_id": owner,
        "table": table_name,
        "columns": column_defs,
        "foreign_keys": foreign_keys,
    }


def get_relationship_graph(
    file_id: str,
    *,
    user_id: Optional[str] = None,
    include_columns: bool = False,
) -> Dict[str, Any]:
    sqlite_path, owner = _locate_sqlite(file_id, user_id)
    base_dir = sqlite_path.parent
    schema_path = base_dir / "ai_spec_schema.json"

    if schema_path.exists():
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
    else:
        schema = None

    if not schema:
        # Fallback: build simple graph from PRAGMA foreign keys
        tables = list_tables(file_id, owner)
        graph: Dict[str, Any] = {"nodes": [], "edges": [], "source": "pragma"}
        for table in tables:
            graph["nodes"].append({"id": f"table:{table}", "type": "Table", "properties": {"name": table}})
        with _connect(sqlite_path) as conn:
            for table in tables:
                fks = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
                for idx, fk in enumerate(fks):
                    target = fk["table"]
                    if target not in tables:
                        continue
                    graph["edges"].append(
                        {
                            "type": "REFERENCES",
                            "from": f"table:{table}",
                            "to": f"table:{target}",
                            "properties": {
                                "from_column": fk["from"],
                                "to_column": fk["to"],
                                "ordinal": idx,
                            },
                        }
                    )
        logger.info("Built relationship graph from PRAGMA", file_id=file_id, user_id=owner)
        return graph

    graph: Dict[str, Any] = {"nodes": [], "edges": [], "source": "ai_spec"}
    column_ids: Dict[str, Dict[str, str]] = {}
    tables = schema.get("tables", [])
    for table in tables:
        name = table.get("table_name")
        table_id = f"table:{name}"
        graph["nodes"].append({"id": table_id, "type": "Table", "properties": {"name": name}})
        column_ids[name] = {}
        if include_columns:
            for column in table.get("columns", []):
                col_name = column.get("column_name")
                col_id = f"column:{name}.{col_name}"
                column_ids[name][col_name] = col_id
                graph["nodes"].append(
                    {
                        "id": col_id,
                        "type": "Column",
                        "properties": {
                            "table": name,
                            "name": col_name,
                            "data_type": column.get("column_dtype"),
                        },
                    }
                )
                graph["edges"].append({"type": "HAS_COLUMN", "from": table_id, "to": col_id, "properties": {}})

        for idx, fk in enumerate(table.get("foreign_keys", [])):
            target = fk.get("to_table")
            if not target:
                continue
            edge_props = {
                "from_columns": fk.get("from_column"),
                "to_columns": fk.get("to_column"),
                "relationship_type": fk.get("relationship_type"),
            }
            graph["edges"].append(
                {
                    "type": "REFERENCES",
                    "from": table_id,
                    "to": f"table:{target}",
                    "properties": edge_props,
                }
            )
            if include_columns:
                from_cols = fk.get("from_column")
                to_cols = fk.get("to_column")
                from_list = from_cols if isinstance(from_cols, list) else [from_cols]
                to_list = to_cols if isinstance(to_cols, list) else [to_cols]
                for ordinal, (fc, tc) in enumerate(zip(from_list, to_list, strict=False)):
                    from_id = column_ids.get(name, {}).get(fc)
                    to_id = column_ids.get(target, {}).get(tc)
                    if from_id and to_id:
                        graph["edges"].append(
                            {
                                "type": "FK_COLUMN",
                                "from": from_id,
                                "to": to_id,
                                "properties": {"ordinal": ordinal, "fk_index": idx},
                            }
                        )
    logger.info("Built relationship graph from ai_spec_schema.json", file_id=file_id, user_id=owner)
    return graph


_WRITE_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "VACUUM", "ATTACH", "REINDEX", "CREATE", "REPLACE"}


def validate_query(
    file_id: str,
    sql: str,
    *,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not sql or not sql.strip():
        raise ValueError("Query is empty")
    sql_stripped = sql.strip()
    upper = sql_stripped.upper()
    if any(upper.startswith(k) for k in _WRITE_KEYWORDS):
        return {
            "is_valid": False,
            "risk_level": "high",
            "error": "Only read-only queries are allowed",
            "query_type": upper.split(" ", 1)[0],
        }

    sqlite_path, owner = _locate_sqlite(file_id, user_id)
    try:
        with _connect(sqlite_path) as conn:
            conn.execute("EXPLAIN QUERY PLAN " + sql_stripped)
        logger.info("Validated query", file_id=file_id, user_id=owner)
        return {"is_valid": True, "risk_level": "low", "query_type": upper.split(" ", 1)[0]}
    except sqlite3.Error as exc:
        logger.warning("Query validation failed", file_id=file_id, user_id=owner, error=str(exc))
        return {"is_valid": False, "risk_level": "medium", "error": str(exc), "query_type": upper.split(" ", 1)[0]}


def _discover_sample_sqlite_entry() -> Tuple[str, str] | None:
    meta_path = settings.storage.data_root / "metadata" / "files.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for file_id, record in payload.items():
        if record.get("file_type") == "sqlite":
            user = record.get("user")
            if user and file_id:
                return user, file_id
    return None


def _run_smoke_tests() -> None:
    sample = _discover_sample_sqlite_entry()
    if not sample:
        print("No sqlite entries found in metadata; skipping smoke tests.")
        return
    user_id, file_id = sample
    print(f"[test] sample sqlite -> user={user_id} file_id={file_id}")

    try:
        dbs = list_databases(user_id)
        print(f"[test] list_databases -> {dbs}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] list_databases error: {exc}")
        return

    try:
        tables = list_tables(file_id, user_id)
        print(f"[test] list_tables -> {tables}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] list_tables error: {exc}")
        return

    details: Optional[Dict[str, Any]] = None
    if tables:
        table = tables[0]
        try:
            sample_rows = sample_data_table(file_id, table, user_id=user_id)
            print(f"[test] sample_data_table rows={len(sample_rows)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[test] sample_data_table error: {exc}")

        try:
            details = get_table_details(file_id, table, user_id=user_id)
            print(f"[test] get_table_details columns={len(details.get('columns', []))}")
        except Exception as exc:  # noqa: BLE001
            print(f"[test] get_table_details error: {exc}")

        if details and details.get("columns"):
            try:
                first_col = details["columns"][0]["name"]
                stats = get_column_stats(file_id, table, first_col, user_id=user_id)
                print(f"[test] get_column_stats distinct={stats.get('distinct_count')}")
            except Exception as exc:  # noqa: BLE001
                print(f"[test] get_column_stats error: {exc}")

    try:
        graph = get_relationship_graph(file_id, user_id=user_id, include_columns=False)
        print(f"[test] relationship graph nodes={len(graph.get('nodes', []))} edges={len(graph.get('edges', []))}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] get_relationship_graph error: {exc}")

    try:
        validation = validate_query(file_id, "SELECT 1;", user_id=user_id)
        print(f"[test] validate_query -> {validation}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] validate_query error: {exc}")


if __name__ == "__main__":
    _run_smoke_tests()
