from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("mcp_csv_tools")


def _artifacts_root() -> Path:
    base = settings.storage.data_root / "artifacts"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _list_parquet_dirs_for_user(user_id: str) -> List[Path]:
    root = _artifacts_root() / user_id
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir() and (p / "data.parquet").exists()]


def list_user_parquet_ids(user_id: str) -> List[str]:
    """
    Return file_ids under artifacts/{user_id}/ that contain a data.parquet.
    """
    dirs = _list_parquet_dirs_for_user(user_id)
    ids = sorted(p.name for p in dirs)
    logger.info("Listed parquet ids for user", user_id=user_id, count=len(ids))
    return ids


def _locate_parquet(file_id: str) -> Path:
    """
    Find the parquet path for a file_id across all users.
    Raises if not found or ambiguous.
    """
    matches = list(_artifacts_root().glob(f"*/{file_id}/data.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet found for file_id={file_id}")
    if len(matches) > 1:
        raise ValueError(f"Multiple parquet files found for file_id={file_id}; please disambiguate by user.")
    return matches[0].resolve()


def _load_dataframe(file_id: str) -> Tuple[pd.DataFrame, Path]:
    parquet_path = _locate_parquet(file_id)
    logger.info("Loading parquet", file_id=file_id, path=str(parquet_path))
    df = pd.read_parquet(parquet_path)
    return df, parquet_path


def data_sample(file_id: str, position: str = "head", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Return top/bottom rows from a parquet dataset identified by file_id.
    """
    position_norm = position.lower()
    if position_norm not in {"head", "tail"}:
        raise ValueError("position must be 'head' or 'tail'")
    df, path = _load_dataframe(file_id)
    sampler = df.head if position_norm == "head" else df.tail
    sample_df = sampler(limit)
    logger.info("Sampled dataframe", file_id=file_id, position=position_norm, rows=len(sample_df), path=str(path))
    return sample_df.to_dict(orient="records")


def get_columns_stats(file_id: str, column_name: str) -> Dict[str, Any]:
    """
    Compute basic stats for a column in the parquet dataset.
    """
    df, path = _load_dataframe(file_id)
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in parquet for file_id={file_id}")

    series = df[column_name]
    total = int(len(series))
    null_mask = series.isnull()
    null_count = int(null_mask.sum())
    nan_count = int(series.apply(lambda v: isinstance(v, float) and math.isnan(v)).sum())
    unique_count = int(series.nunique(dropna=True))

    value_counts = series.value_counts(dropna=True)
    top_values = []
    for value, count in value_counts.head(10).items():
        pct = float(count) / total if total else 0.0
        # Ensure JSON serializable
        safe_value: Any = value
        try:
            json.dumps(value)
        except TypeError:
            safe_value = str(value)
        top_values.append({"value": safe_value, "count": int(count), "pct": pct})

    stats: Dict[str, Any] = {
        "file_id": file_id,
        "column": column_name,
        "dtype": str(series.dtype),
        "total_rows": total,
        "null_count": null_count,
        "nan_count": nan_count,
        "unique_count": unique_count,
        "top_values": top_values,
        "path": str(path),
    }

    if is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        def _safe(val: Any) -> float | None:
            if pd.isna(val):
                return None
            try:
                if isinstance(val, float) and math.isnan(val):
                    return None
            except Exception:
                return None
            return float(val)
        stats["numeric_summary"] = {
            "min": _safe(numeric_series.min(skipna=True)),
            "max": _safe(numeric_series.max(skipna=True)),
            "mean": _safe(numeric_series.mean(skipna=True)),
            "std": _safe(numeric_series.std(skipna=True)),
        }

    logger.info("Computed column stats", file_id=file_id, column=column_name)
    return stats


def verify_function(file_id: str, func: Callable[[pd.DataFrame], Any]) -> Dict[str, Any]:
    """
    Run a callable against the dataframe; return success/error info.
    """
    if not callable(func):
        raise TypeError("func must be callable")
    df, path = _load_dataframe(file_id)
    try:
        func(df)
        logger.info("verify_function succeeded", file_id=file_id, path=str(path))
        return {"success": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.warning("verify_function failed", file_id=file_id, error=str(exc), path=str(path))
        return {"success": False, "error": str(exc)}


def _discover_sample_csv_entry() -> Tuple[str, str] | None:
    """
    Try to find a CSV-derived parquet from metadata snapshots for quick testing.
    """
    meta_path = settings.storage.data_root / "metadata" / "files.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for file_id, record in payload.items():
        if record.get("file_type") == "csv":
            user = record.get("user")
            if user and file_id:
                return user, file_id
    return None


def _run_smoke_tests() -> None:
    """
    Lightweight smoke tests when running this module directly.
    """
    sample = _discover_sample_csv_entry()
    if not sample:
        print("No CSV entries found in metadata; skipping smoke tests.")
        return
    user_id, file_id = sample
    print(f"[test] found sample csv -> user={user_id} file_id={file_id}")

    try:
        ids = list_user_parquet_ids(user_id)
        print(f"[test] list_user_parquet_ids -> {ids}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] list_user_parquet_ids error: {exc}")
        return

    try:
        sample_rows = data_sample(file_id, "head")
        print(f"[test] data_sample rows={len(sample_rows)} first_row_keys={list(sample_rows[0].keys()) if sample_rows else []}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] data_sample error: {exc}")
        return

    try:
        df, _ = _load_dataframe(file_id)
        first_col = df.columns[0]
        stats = get_columns_stats(file_id, first_col)
        print(f"[test] get_columns_stats({first_col}) unique={stats.get('unique_count')} nulls={stats.get('null_count')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] get_columns_stats error: {exc}")
        return

    try:
        result = verify_function(file_id, lambda frame: frame.head(1))
        print(f"[test] verify_function success={result['success']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] verify_function error: {exc}")


if __name__ == "__main__":
    _run_smoke_tests()
