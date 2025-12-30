from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ColumnSpec(BaseModel):
    column_name: str
    column_dtype: str
    column_comment: str
    column_business_interpretation: str
    importance_label: Literal["high", "medium", "low"]


class ForeignKeySpec(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: Literal["many-to-one", "one-to-many", "one-to-one"]
    foreign_key_comment: str
    foreign_key_business_interpretation: str


class TableSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    table_name: str
    primary_key: str | list[str]
    foreign_keys: list[ForeignKeySpec] = Field(default_factory=list)
    columns: list[ColumnSpec]
    table_name_comment: str
    primary_key_comment: str
    table_name_business_interpretations: str
    primary_key_business_interpretations: str


class DatabaseSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    last_updated: str
    database_type: str
    tables: list[TableSpec]


class SchemaReview(BaseModel):
    verdict: Literal["pass", "revise"]
    feedback: str
    issues: list[str] = Field(default_factory=list)
