from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

from agents import Agent, Runner, TResponseInputItem
from tqdm import tqdm

from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.db.schema_assembler_models import DatabaseSchema, SchemaReview, TableSpec
from backend.module.pipelines.db.schema_assembler_prompts import (
    build_balanced_global_judge_prompt,
    build_table_judge_prompt,
    build_table_prompt,
)
from backend.module.pipelines.db.schema_assembler_utils import (
    current_utc_timestamp,
    fetch_table_samples,
    introspect_sqlite_schema,
    load_example_schema,
    load_schema_spec_definition,
    load_schema_sql,
    validate_against_sql_definitions,
)
from backend.module.providers.oai_agents import close_client, create_responses_model

logger = LoggingInterceptor("pipelines_db_schema_assembler")


async def assemble_schema(
    sqlite_path: Path,
    schema_sql_path: Path,
    output_path: Path,
    *,
    database_id: Optional[str] = None,
    provider: Optional[str] = None,
    model_profile: Optional[str] = None,
    sample_limit: int = 3,
    max_attempts: int = 5,
    business_context: str | None = None,
) -> DatabaseSchema:
    """
    Generate a validated AI spec schema JSON using OpenAI Agents.

    Args:
        sqlite_path: Path to the SQLite database file.
        schema_sql_path: Path to the schema.sql file describing the DB.
        output_path: Where to write the resulting ai_spec_schema.json.
        database_id: Optional identifier; defaults to sqlite file stem.
        provider: Optional provider override ("openai" or "azure").
        model_profile: Optional profile name (falls back to configured default).
        sample_limit: Rows per table to include as samples.
        max_attempts: Max generator/judge iterations before failing.
        business_context: Optional extra context for the generator (not the judge).
    """
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {sqlite_path}")
    if not schema_sql_path.exists():
        raise FileNotFoundError(f"Schema SQL not found at {schema_sql_path}")

    db_id = database_id or sqlite_path.stem
    logger.info(
        "Starting schema assembly",
        database_id=db_id,
        provider=provider or "default",
        model_profile=model_profile or "default",
        sample_limit=sample_limit,
        max_attempts=max_attempts,
    )

    schema_sql = load_schema_sql(schema_sql_path)
    schema_spec_definition = load_schema_spec_definition()
    example_schema = load_example_schema()
    samples = fetch_table_samples(sqlite_path, limit=sample_limit)
    table_definitions = introspect_sqlite_schema(sqlite_path)
    target_last_updated = current_utc_timestamp()
    expected_tables = sorted(table_definitions.keys())

    print(f"[schema] starting assembly for {db_id}")
    model, client = create_responses_model(profile=model_profile, provider=provider)

    table_agent = Agent(
        name="schema_table_agent",
        instructions=(
            "Generate a single table specification following schema_spec.json TableSpec. "
            "Do not output other tables."
        ),
        output_type=TableSpec,
        model=model,
    )
    judge_agent = Agent(
        name="schema_validator_judge",
        instructions=(
            "Validate generated schema JSON comments/business interpretations against SQL + samples. "
            "Structural coverage is handled separately."
        ),
        output_type=SchemaReview,
        model=model,
    )

    table_feedback: dict[str, str] = {}
    table_last_spec: dict[str, TableSpec] = {}
    accepted_tables: dict[str, TableSpec] = {}
    pending_tables = set(expected_tables)

    candidate: DatabaseSchema | None = None

    try:
        for attempt in range(1, max_attempts + 1):
            if not pending_tables:
                break
            print(f"[schema] attempt {attempt}/{max_attempts} (pending: {len(pending_tables)})")
            logger.info("Generating schema attempt", attempt=attempt, pending=len(pending_tables))

            with tqdm(total=len(pending_tables), desc="tables", unit="table") as pbar:
                for idx, table_name in enumerate(list(pending_tables), start=1):
                    logger.info("Generating table spec", table=table_name, index=idx, total=len(pending_tables))
                    previous_spec_json = (
                        table_last_spec[table_name].model_dump_json(indent=2)
                        if table_name in table_last_spec
                        else None
                    )
                    prompt = build_table_prompt(
                        database_id=db_id,
                        table_name=table_name,
                        table_definition=table_definitions[table_name],
                        sample_rows=samples.get(table_name, []),
                        schema_spec_definition=schema_spec_definition,
                        example_schema=example_schema,
                        business_context=business_context,
                        prior_feedback=table_feedback.get(table_name),
                        previous_table_spec_json=previous_spec_json,
                        attempt_number=attempt,
                        max_attempts=max_attempts,
                    )
                    input_items: List[TResponseInputItem] = [{"role": "user", "content": prompt}]
                    table_result = await Runner.run(table_agent, input_items)
                    table_spec: TableSpec = table_result.final_output
                    table_last_spec[table_name] = table_spec

                    validation_issues = validate_against_sql_definitions(
                        [table_spec], {table_name: table_definitions[table_name]}
                    )
                    if validation_issues:
                        feedback_text = "; ".join(validation_issues)
                        table_feedback[table_name] = feedback_text
                        logger.warning(
                            "Table structural validation failed",
                            table=table_name,
                            issues=validation_issues,
                            attempt=attempt,
                        )
                        print(f"[schema] {table_name}: structural issues -> {feedback_text}")
                        pbar.update(1)
                        continue

                    judge_prompt = build_table_judge_prompt(
                        database_id=db_id,
                        table_name=table_name,
                        schema_sql=schema_sql,
                        samples=samples.get(table_name, []),
                        candidate_table_json=table_spec.model_dump_json(indent=2),
                        attempt_number=attempt,
                        max_attempts=max_attempts,
                    )
                    judge_result = await Runner.run(judge_agent, [{"role": "user", "content": judge_prompt}])
                    review: SchemaReview = judge_result.final_output

                    if review.verdict == "pass":
                        accepted_tables[table_name] = table_spec
                        pending_tables.remove(table_name)
                        table_feedback.pop(table_name, None)
                        print(f"[schema] {table_name}: judge passed ✅")
                        logger.info("Table judge passed", table=table_name, attempt=attempt)
                    else:
                        feedback_entry = review.feedback
                        if review.issues:
                            feedback_entry = f"{feedback_entry}\nIssues: " + "; ".join(review.issues)
                        table_feedback[table_name] = feedback_entry
                        logger.info("Table judge requested revision", table=table_name, attempt=attempt, feedback=feedback_entry)
                        print(f"[schema] {table_name}: judge revision -> {feedback_entry}")
                    pbar.update(1)

        # Build best-effort candidate using accepted + latest attempts for pending
        if pending_tables:
            for t in pending_tables:
                if t in table_last_spec:
                    accepted_tables[t] = table_last_spec[t]
            logger.warning("Proceeding with best-effort schema despite pending tables", pending=list(pending_tables))

        table_specs_sorted: List[TableSpec] = [accepted_tables[name] for name in sorted(accepted_tables.keys())]
        candidate = DatabaseSchema(
            last_updated=target_last_updated,
            database_type="SQLite",
            tables=table_specs_sorted,
        )

        candidate_json = candidate.model_dump_json(indent=2)
        judge_prompt = build_balanced_global_judge_prompt(
            database_id=db_id,
            schema_sql=schema_sql,
            samples=samples,
            candidate_schema_json=candidate_json,
        )
        judge_result = await Runner.run(judge_agent, [{"role": "user", "content": judge_prompt}])
        review: SchemaReview = judge_result.final_output

        if review.verdict != "pass":
            feedback_entry = review.feedback
            if review.issues:
                feedback_entry = f"{feedback_entry}\nIssues: " + "; ".join(review.issues)
            logger.warning("Global judge reported issues; saving best-effort schema", feedback=feedback_entry)
        else:
            print("[schema] global judge passed ✅")
            logger.info("Schema passed global judge")

        save_schema(candidate, output_path)
        logger.info("Schema saved", output_path=str(output_path))
        return candidate
    finally:
        await close_client(client)


def save_schema(schema: DatabaseSchema, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(schema.model_dump_json(indent=2), encoding="utf-8")


async def main() -> None:
    raise SystemExit(
        "Use schema_assembler_test.py or import assemble_schema directly; no CLI defaults are provided."
    )


if __name__ == "__main__":
    asyncio.run(main())
