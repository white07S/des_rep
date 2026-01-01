from __future__ import annotations

import asyncio
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.module.codex_agents.framework.assembler import SNAPSHOT_ROOT
from backend.module.codex_agents.framework.utils import codex_safe_schema, get_agent_config, parse_structured_response, run_agent_once
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("agents_viz_agent")

AGENT_NAME = "viz_agent"

ResponseType = Literal["viz-function", "viz-explore"]


class VizAgentResponse(BaseModel):
    """Visualization agent response supporting function generation and exploration."""

    misleading: bool = Field(description="True if request is misleading or cannot be answered")
    response_type: Optional[ResponseType] = Field(
        default=None,
        description="Type of response: 'viz-function' for visualization, 'viz-explore' for exploration. None if misleading",
    )
    reasoning: List[str] = Field(description="Step-by-step reasoning for the response")
    python_function: str = Field(
        default="",
        description="Python function that generates ECharts specs. Only populated when response_type='viz-function'",
    )
    chart_specs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Optional example ECharts specifications. Can be empty list",
    )
    text: str = Field(
        default="",
        description="Exploratory text response about the data. Only populated when response_type='viz-explore'",
    )


def get_viz_response_json_schema() -> Dict[str, Any]:
    """Generate Codex-safe JSON Schema for VizAgentResponse."""
    return codex_safe_schema(VizAgentResponse)


def parse_viz_response(text: str) -> VizAgentResponse:
    """Parse the final agent message as VizAgentResponse, tolerating extra text."""
    return parse_structured_response(text, VizAgentResponse)


def prepare_dataframe_metadata(df_head: List[Dict[str, Any]], df_tail: List[Dict[str, Any]], columns: List[str], shape: List[int]) -> Dict[str, Any]:
    """Prepare DataFrame metadata for the visualization agent."""
    dtypes: Dict[str, str] = {}
    for col in columns:
        sample_val: Any = None
        for row in df_head + df_tail:
            if col in row and row[col] is not None:
                sample_val = row[col]
                break
        if sample_val is None:
            dtypes[col] = "unknown"
        elif isinstance(sample_val, bool):
            dtypes[col] = "bool"
        elif isinstance(sample_val, int):
            dtypes[col] = "int"
        elif isinstance(sample_val, float):
            dtypes[col] = "float"
        elif isinstance(sample_val, str):
            if any(char in sample_val for char in ["-", "/", ":"]):
                dtypes[col] = "datetime"
            else:
                dtypes[col] = "str"
        else:
            dtypes[col] = "object"

    return {
        "columns": columns,
        "dtypes": dtypes,
        "shape": shape,
        "head_sample": df_head,
        "tail_sample": df_tail,
    }


def create_test_dataframe_from_metadata(metadata: Dict[str, Any]) -> Any:
    """Create a pandas DataFrame from metadata for testing the generated function."""
    try:
        import pandas as pd
        import numpy as np  # noqa: F401

        all_samples = metadata.get("head_sample", []) + metadata.get("tail_sample", [])
        if not all_samples:
            return pd.DataFrame(columns=metadata.get("columns", []))

        df = pd.DataFrame(all_samples)
        dtypes = metadata.get("dtypes", {})
        for col, dtype in dtypes.items():
            if col not in df.columns:
                continue
            if dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        target_rows = metadata.get("shape", [len(df), len(df.columns)])[0]
        if len(df) < min(target_rows, 100) and len(df) > 0:
            repeats = min(100, target_rows) // len(df) + 1
            df = pd.concat([df] * repeats, ignore_index=True)[: min(100, target_rows)]
        return df
    except ImportError as exc:
        logger.warning("pandas not installed, cannot create test DataFrame", error=str(exc))
        return None
    except Exception as exc:
        logger.error("Failed to create test DataFrame", error=str(exc))
        return None


def execute_visualization_function(function_code: str, test_dataframe: Any, safe_mode: bool = True) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """Safely execute the generated Python function with test data."""
    if test_dataframe is None:
        return None
    try:
        safe_globals: Dict[str, Any] = {"__builtins__": __builtins__, "json": json, "datetime": datetime}
        if not safe_mode:
            safe_globals = dict(__builtins__=__builtins__)

        try:
            import pandas as pd
            import numpy as np

            safe_globals.update({"pd": pd, "pandas": pd, "np": np, "numpy": np})
        except ImportError as exc:
            logger.warning("Optional dependencies missing for viz function execution", error=str(exc))

        local_namespace: Dict[str, Any] = {}
        exec(function_code, safe_globals, local_namespace)

        func = next((obj for name, obj in local_namespace.items() if callable(obj) and not name.startswith("_")), None)
        if func is None:
            logger.error("No callable function found in generated code")
            return None

        result = func(test_dataframe)
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and all(isinstance(item, dict) for item in result):
            return result
        logger.error("Invalid result type from generated function", result_type=str(type(result)))
        return None
    except SyntaxError as exc:
        logger.error("Syntax error in generated code", error=str(exc), lineno=exc.lineno, text=exc.text)
        return None
    except Exception as exc:
        logger.error("Error executing generated function", error=str(exc), traceback=traceback.format_exc())
        return None


def save_echarts_spec(spec: Union[Dict[str, Any], List[Dict[str, Any]]], output_path: Path, pretty: bool = True) -> bool:
    """Save the ECharts specification to a JSON file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            if pretty:
                json.dump(spec, handle, indent=2, ensure_ascii=False)
            else:
                json.dump(spec, handle, ensure_ascii=False)
        logger.info("Saved ECharts spec", path=str(output_path), bytes=output_path.stat().st_size)
        return True
    except Exception as exc:
        logger.error("Failed to save ECharts spec", path=str(output_path), error=str(exc))
        return False


def test_generated_function(function_code: str, metadata: Dict[str, Any], output_dir: Optional[Union[str, Path]] = None, verbose: bool = True) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Complete testing pipeline: create DataFrame, execute function, save spec.
    """
    target_dir = _default_output_dir() if output_dir is None else Path(output_dir)
    if verbose:
        print("\n" + "=" * 60)
        print("TESTING GENERATED VISUALIZATION FUNCTION")
        print("=" * 60)

    if verbose:
        print("\n1) Creating test DataFrame from metadata...")
    test_df = create_test_dataframe_from_metadata(metadata)
    if test_df is None:
        return None
    if verbose:
        print(f"   Created DataFrame with shape: {getattr(test_df, 'shape', None)}")
        print(f"   Columns: {list(getattr(test_df, 'columns', []))}")

    if verbose:
        print("\n2) Executing generated function...")
    spec = execute_visualization_function(function_code, test_df)
    if spec is None:
        return None
    if verbose:
        print("   Function executed successfully")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = target_dir / f"echarts_spec_{timestamp}.json"
    function_path = target_dir / f"viz_function_{timestamp}.py"

    if verbose:
        print("\n3) Saving ECharts specification...")
    saved = save_echarts_spec(spec, output_path)
    if saved:
        try:
            function_path.write_text(function_code, encoding="utf-8")
            logger.info("Saved generated function", path=str(function_path))
            if verbose:
                print(f"   Function code saved to: {function_path}")
        except Exception as exc:
            logger.warning("Failed to save generated function code", path=str(function_path), error=str(exc))
    if verbose:
        print("\n" + "=" * 60)
        print("TESTING COMPLETE")
        print("=" * 60)
    return spec


def _default_output_dir() -> Path:
    """Return default output directory under the viz agent snapshot."""
    return SNAPSHOT_ROOT / AGENT_NAME / "viz_output"


async def run_demo() -> None:
    """
    Run a demo prompt against the assembled viz_agent, executing the generated function if provided.
    """
    agent_config = get_agent_config(AGENT_NAME)
    prompt = "Generate a visualization for quarterly revenue trends with regions and annotate max revenue point."

    df_metadata = prepare_dataframe_metadata(
        df_head=[
            {"quarter": "2024-Q1", "region": "North", "revenue": 1200000.0},
            {"quarter": "2024-Q2", "region": "South", "revenue": 1310000.0},
            {"quarter": "2024-Q3", "region": "East", "revenue": 1185000.0},
        ],
        df_tail=[
            {"quarter": "2024-Q4", "region": "West", "revenue": 1540000.0},
            {"quarter": "2024-Q4", "region": "North", "revenue": 1615000.0},
            {"quarter": "2024-Q4", "region": "South", "revenue": 1432000.0},
        ],
        columns=["quarter", "region", "revenue"],
        shape=[16, 3],
    )

    output_schema = get_viz_response_json_schema()
    result = await run_agent_once(
        AGENT_NAME,
        prompt,
        output_schema=output_schema,
        on_event=lambda e: print(json.dumps(e, ensure_ascii=False)),
    )

    if not result.final_message:
        logger.error("No agent message received from viz agent session")
        return

    try:
        response = parse_viz_response(result.final_message)
    except Exception as exc:
        logger.error("Failed to parse viz agent response", error=str(exc))
        return

    logger.info(
        "Parsed viz agent response",
        misleading=response.misleading,
        response_type=response.response_type,
        reasoning=" | ".join(response.reasoning),
    )

    if not response.misleading and response.response_type == "viz-function" and response.python_function:
        test_generated_function(
            function_code=response.python_function,
            metadata=df_metadata,
            output_dir=agent_config.agent_folder / "viz_output",
            verbose=True,
        )


if __name__ == "__main__":
    asyncio.run(run_demo())
