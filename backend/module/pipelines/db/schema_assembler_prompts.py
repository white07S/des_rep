from __future__ import annotations

import json
from typing import Any, Dict, List

from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("pipelines_db_schema_assembler_prompts")


def build_table_prompt(
    database_id: str,
    table_name: str,
    table_definition: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
    schema_spec_definition: str,
    example_schema: str,
    business_context: str | None = None,
    prior_feedback: str | None = None,
    previous_table_spec_json: str | None = None,
    attempt_number: int | None = None,
    max_attempts: int | None = None,
) -> str:
    pk_list = table_definition["primary_keys"]
    pk_text = pk_list if pk_list else []
    fk_text = json.dumps(table_definition["foreign_keys"], indent=2)
    columns_text = json.dumps(table_definition["columns"], indent=2)
    samples_text = json.dumps(sample_rows, indent=2)
    feedback_text = f"\nPrior feedback: {prior_feedback}" if prior_feedback else ""
    previous_spec_text = (
        f"\nPrevious attempt for this table (JSON):\n{previous_table_spec_json}"
        if previous_table_spec_json
        else ""
    )
    business_context_text = (
        f"\nAdditional business context supplied by the user (use to enrich interpretations):\n{business_context.strip()}"
        if business_context
        else "\nNo additional business context was provided."
    )
    attempt_text = (
        f"\nCurrent attempt: {attempt_number}/{max_attempts}" if attempt_number and max_attempts else ""
    )
    pk_instruction = (
        "No primary key declared in SQL; set primary_key to an empty list [] and explain in comments."
        if not pk_list
        else "Use the primary_key values as declared in SQL."
    )

    logger.info("Building table prompt", table=table_name, has_feedback=bool(prior_feedback))
    return (
        f"You are the Schema Table Agent. Produce a JSON object for one table that matches the TableSpec shape from schema_spec.json."
        f"\n\nDatabase id: {database_id}"
        f"\nTable: {table_name}"
        f"\nColumn definitions (from PRAGMA):\n{columns_text}"
        f"\n\nPrimary key columns: {pk_text}"
        f"\nForeign keys (from PRAGMA):\n{fk_text}"
        f"\n\nSample rows (top rows from SQLite):\n{samples_text}"
        f"\n\nSchema spec reference (for structure and field names):\n{schema_spec_definition}"
        f"\nExample schema reference:\n{example_schema}"
        f"{business_context_text}"
        f"{previous_spec_text}"
        f"{attempt_text}"
        f"\n\nRequirements:"
        f"\n- Output only the JSON for this table (TableSpec), not the full database."
        f"\n- Keep column and table names exactly as in SQL (case-sensitive as shown)."
        f"\n- {pk_instruction}"
        f"\n- Use column_dtype exactly as provided in PRAGMA; do not change or suggest alternative types. If a value looks numeric but is stored as text, keep the dtype and note that it is a text-coded quantity in the comment."
        f"\n- Align comments with sample values; if encoding is unclear, describe the code pattern using observed values (e.g., '1', '2') without inventing a new scheme."
        f"\n- For categorical codes (including gender/status/etc.), describe what the codes represent only if clearly evident from samples/schema; otherwise, state they are codes and avoid inventing meanings."
        f"\n- For foreign keys, keep relationship_type as many-to-one unless evidence shows otherwise."
        f"\n- Column comments must be technical and grounded; business interpretations must be distinct and about analytical or operational use."
        f"\n- Importance labels: high for primary/foreign keys and pivotal dates/ids; high for status codes that drive flows; medium for contact fields and descriptors used in operations/marketing; low for auxiliary text-only fields."
        f"\n- Do not invent columns or tables."
        f"{feedback_text}"
    ).strip()


def build_judge_prompt(
    database_id: str,
    schema_sql: str,
    samples: Dict[str, List[Dict[str, Any]]],
    candidate_schema_json: str,
) -> str:
    samples_text = json.dumps(samples, indent=2)
    logger.info("Building judge prompt", database_id=database_id)
    return (
        f"You are the Schema Validator Judge. Review only the comments and business interpretations for {database_id}."
        f"\n\nSQL schema:\n{schema_sql}"
        f"\n\nSamples (top rows per table):\n{samples_text}"
        f"\n\nCandidate schema JSON:\n{candidate_schema_json}"
        f"\n\nChecks:"
        f"\n1) Column comments are meaningful, concise, and grounded in schema + samples."
        f"\n2) Business interpretations must be distinct from comments and focus on analytical or operational use, written for multiple perspectives and detailed."
        f"\n3) Importance labels fit: primary/foreign keys high; key dates and ids high; descriptive join fields medium; auxiliary text low."
        f"\n\nRespond with verdict='pass' if these checks are satisfied. Otherwise set verdict='revise' and provide concise, actionable feedback referencing specific tables/columns."
    ).strip()


def build_table_judge_prompt(
    database_id: str,
    table_name: str,
    schema_sql: str,
    samples: List[Dict[str, Any]],
    candidate_table_json: str,
    attempt_number: int | None = None,
    max_attempts: int | None = None,
) -> str:
    samples_text = json.dumps(samples, indent=2)
    logger.info("Building table judge prompt", table=table_name)
    attempt_text = (
        f"This is attempt {attempt_number}/{max_attempts}. If only minor stylistic clarity or optional elaboration issues remain, respond with verdict='pass' and list suggestions in issues; reserve verdict='revise' for material problems (wrong linkage, misleading comments vs samples/SQL, or clearly mis-set importance labels). Do not request column_dtype changes—accept PRAGMA types even if text-coded numeric values appear."
        if attempt_number and max_attempts
        else "If issues are minor or stylistic, respond with verdict='pass' and include suggestions in issues. Use verdict='revise' only for material problems (misaligned with SQL/samples or clearly incorrect importance labels). Do not request column_dtype changes—accept PRAGMA types even if text-coded numeric values appear."
    )
    return (
        f"You are the Schema Validator Judge for a single table in database {database_id}. "
        f"Review comments and business interpretations for table {table_name}."
        f"\n\nSQL schema:\n{schema_sql}"
        f"\n\nSamples (top rows for this table):\n{samples_text}"
        f"\n\nCandidate table JSON:\n{candidate_table_json}"
        f"\n\n{attempt_text}"
        f"\n\nChecks:"
        f"\n1) Column comments are meaningful, concise, and grounded in schema + samples."
        f"\n2) Business interpretations must be distinct from comments and focus on analytical or operational use, written for multiple perspectives and detailed."
        f"\n3) Importance labels fit: primary/foreign keys high; key dates and ids high; descriptive join fields medium; auxiliary text low."
        f"\n4) Accept column_dtype exactly as PRAGMA provides; do not request dtype changes even if values look numeric."
        f"\n5) For coded categorical fields (e.g., gender/status), accept code descriptions; do not require mapping to different encodings unless schema shows it."
        f"\n6) If you notice plausible relationships not captured in PRAGMA, list them as recommendations in issues; do not fail the table solely for proposing them."
        f"\n\nRespond with verdict='pass' if these checks are satisfied. Otherwise set verdict='revise' and provide concise, actionable feedback referencing specific columns."
    ).strip()


def build_balanced_global_judge_prompt(
    database_id: str,
    schema_sql: str,
    samples: Dict[str, List[Dict[str, Any]]],
    candidate_schema_json: str,
) -> str:
    samples_text = json.dumps(samples, indent=2)
    logger.info("Building balanced global judge prompt", database_id=database_id)
    return (
        f"You are the final Schema Validator Judge for database {database_id}. The table-level judges already passed."
        f"\n\nSQL schema:\n{schema_sql}"
        f"\n\nSamples (top rows per table):\n{samples_text}"
        f"\n\nCandidate schema JSON:\n{candidate_schema_json}"
        f"\n\nChecks (be pragmatic):"
        f"\n1) Tables/columns align with SQL; no invented structures."
        f"\n2) Comments/business interpretations are grounded and distinct; minor wording issues should not block approval."
        f"\n3) Importance labels broadly align with guidance; only fail if clearly wrong (e.g., primary keys not high)."
        f"\n4) Accept column_dtype exactly as PRAGMA provides; do not request dtype changes even if values look numeric."
        f"\n5) If you see plausible relationships not captured in PRAGMA, list them as recommendations in issues; do not fail the schema solely for proposing them."
        f"\n\nIf only minor or stylistic improvements remain, respond with verdict='pass' and include suggestions in issues. "
        f"Use verdict='revise' only for material correctness gaps."
    ).strip()


__all__ = ["build_table_prompt", "build_judge_prompt", "build_table_judge_prompt", "build_balanced_global_judge_prompt"]
