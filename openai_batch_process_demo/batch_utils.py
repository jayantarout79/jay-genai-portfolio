"""
Utility helpers for working with the OpenAI Batch API demo.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables on import so the client picks up the API key.
load_dotenv()


def ensure_data_dirs() -> Tuple[Path, Path]:
    """
    Ensure the data directories exist for inputs and outputs.
    Returns the created/verified data and sample_data paths.
    """
    data_dir = Path("data")
    sample_dir = Path("sample_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, sample_dir


def get_client() -> OpenAI:
    """
    Build an OpenAI client using the API key from the environment.
    Raises a ValueError if the key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")
    client = OpenAI()
    print("OpenAI client initialized.")
    return client


def load_feedback_csv(path: Path) -> pd.DataFrame:
    """
    Load a feedback CSV and validate required columns.
    """
    if not path.exists():
        raise ValueError(f"CSV file not found at {path}")
    df = pd.read_csv(path)
    required_cols = {"id", "customer_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
    print(f"Loaded CSV with {len(df)} rows from {path}.")
    return df


def _build_prompt(feedback: str) -> str:
    """
    Build the prompt string for the Responses API call.
    """
    return (
        "You are an expert customer support data analyst. Given the customer feedback "
        "text, classify it.\n\n"
        "Return STRICTLY a JSON object with keys: sentiment (positive/neutral/negative), "
        "category (product/pricing/support/experience/other), short_summary (max 25 words).\n\n"
        f'Customer feedback: "{feedback}"'
    )


def build_batch_jsonl(df: pd.DataFrame, model: str, output_path: Path) -> None:
    """
    Build the Batch API JSONL input file for all rows in the DataFrame.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx, row in df.iterrows():
        feedback_text = str(row.get("customer_text", "")).strip()
        if not feedback_text:
            print(f"Skipping empty feedback at row {idx}")
            continue
        row_id = row.get("id", idx)
        custom_id = f"feedback-{row_id}"
        body = {
            "model": model,
            "input": _build_prompt(feedback_text),
        }
        payload = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        lines.append(json.dumps(payload))
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} batch requests to {output_path}.")


def upload_batch_file(client: OpenAI, jsonl_path: Path) -> str:
    """
    Upload the JSONL file to OpenAI for batch processing.
    Returns the uploaded file ID.
    """
    if not jsonl_path.exists():
        raise ValueError(f"JSONL file not found at {jsonl_path}")
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    file_id = uploaded.id
    print(f"Uploaded batch file with id={file_id}.")
    return file_id


def create_batch_job(client: OpenAI, file_id: str, completion_window: str = "24h") -> str:
    """
    Create a Batch job using the uploaded input file and return the batch ID.
    """
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    print(f"Created batch job with id={batch.id}.")
    return batch.id


def get_batch_status(client: OpenAI, batch_id: str) -> Dict[str, Any]:
    """
    Retrieve status information for a Batch job.
    """
    batch = client.batches.retrieve(batch_id)
    raw_counts = getattr(batch, "request_counts", {}) or {}
    if isinstance(raw_counts, dict):
        request_counts = raw_counts
    else:
        # The SDK may return an object; convert it to a simple dict.
        request_counts = {}
        for key in ["total", "completed", "failed"]:
            if hasattr(raw_counts, key):
                request_counts[key] = getattr(raw_counts, key)
    status_info = {
        "id": batch.id,
        "status": batch.status,
        "request_counts": request_counts,
        "input_file_id": getattr(batch, "input_file_id", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "created_at": getattr(batch, "created_at", None),
        "completed_at": getattr(batch, "completed_at", None),
        "expires_at": getattr(batch, "expires_at", None),
    }
    print(f"Batch {batch_id} status: {status_info}")
    return status_info


def download_batch_output_file(client: OpenAI, output_file_id: str, dest_path: Path) -> None:
    """
    Download the batch output file and save it locally.
    """
    response = client.files.content(output_file_id)
    content: Any
    # New SDK returns a FileContent object with .text or .read().
    if hasattr(response, "text"):
        content = response.text
    elif hasattr(response, "read"):
        content = response.read().decode("utf-8")
    else:
        content = str(response)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(content, encoding="utf-8")
    print(f"Downloaded output file {output_file_id} to {dest_path}.")


def parse_batch_output(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Parse the batch output JSONL file into structured rows.
    """
    if not jsonl_path.exists():
        raise ValueError(f"Output JSONL not found at {jsonl_path}")
    parsed: List[Dict[str, Any]] = []
    failed_lines = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                custom_id = record.get("custom_id", "")
                response_section = record.get("response", {}) or {}
                # Some SDKs nest body under response["body"], others put output directly under response.
                body = response_section.get("body", response_section)
                output = body.get("output", response_section.get("output", []))

                text = ""
                if isinstance(output, list) and output:
                    content_blocks = output[0].get("content", [])
                    if content_blocks and isinstance(content_blocks, list):
                        # Find the first content block with text.
                        for block in content_blocks:
                            if isinstance(block, dict) and "text" in block:
                                text = block.get("text", "")
                                break

                # Remove common markdown fences that models sometimes include.
                text = text.strip()
                fence_match = re.match(r"```[a-zA-Z]*\n(.*)\n```", text, flags=re.DOTALL)
                if fence_match:
                    text = fence_match.group(1).strip()

                data = json.loads(text)
                parsed.append(
                    {
                        "custom_id": custom_id,
                        "sentiment": data.get("sentiment"),
                        "category": data.get("category"),
                        "short_summary": data.get("short_summary"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failed_lines += 1
                print(f"Failed to parse line {line_num}: {exc}")
                continue
    print(f"Parsed {len(parsed)} results from batch output. Failed lines: {failed_lines}")
    return parsed


def merge_results_with_input(df: pd.DataFrame, parsed_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge parsed model outputs back onto the original DataFrame using custom_id.
    """
    result_df = pd.DataFrame(parsed_results)
    merged = df.copy()
    merged["id"] = merged["id"].astype(str)

    if result_df.empty:
        # Ensure expected columns exist even if no results parsed.
        for col in ["sentiment", "category", "short_summary"]:
            if col not in merged.columns:
                merged[col] = None
        print("No parsed results to merge; returning original data with empty result columns.")
        return merged

    # Extract original id from custom_id (format: feedback-<id>)
    result_df["id"] = result_df["custom_id"].str.replace("feedback-", "", regex=False)
    merged = merged.merge(result_df.drop(columns=["custom_id"]), on="id", how="left")
    print(f"Merged results onto DataFrame; total rows: {len(merged)}.")
    return merged
