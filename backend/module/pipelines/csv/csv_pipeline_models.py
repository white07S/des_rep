from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CSVPipelineError(Exception):
    """Base exception for CSV pipeline errors."""


class CSVReadError(CSVPipelineError):
    """Raised when loading or type coercion fails."""


class CSVWriteError(CSVPipelineError):
    """Raised when persisting outputs fails."""


class CSVConversionConfig(BaseModel):
    """Configuration for converting a CSV file to an Arrow/Pandas representation."""

    model_config = ConfigDict(extra="forbid")

    input_path: Path
    output_path: Optional[Path] = None
    dtype_overrides: Dict[str, str] = Field(default_factory=dict)
    auto_cast_dates: bool = True
    sample_size: int = -1

    @model_validator(mode="after")
    def _validate_sample_size(self) -> "CSVConversionConfig":
        if self.sample_size == 0 or self.sample_size < -1:
            raise ValueError("sample_size must be -1 (auto) or a positive integer")
        return self


class CSVConversionResult(BaseModel):
    """Converted data and metadata for downstream use."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dataframe: pd.DataFrame
    arrow_table: pa.Table
    output_path: Path
    row_count: int
    column_types: Dict[str, str]


__all__ = [
    "CSVConversionConfig",
    "CSVConversionResult",
    "CSVPipelineError",
    "CSVReadError",
    "CSVWriteError",
]
