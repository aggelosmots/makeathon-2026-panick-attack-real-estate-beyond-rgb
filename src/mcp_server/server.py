from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

from src.common_config import DATA_ROOT, env_bool, env_int, env_str

MCP_HOST = env_str("MCP_HOST", "0.0.0.0")
MCP_PORT = env_int("MCP_PORT", 8000)
ALLOW_WRITE_TO_DATA = env_bool("ALLOW_WRITE_TO_DATA", False)

mcp = FastMCP(
    "local-data-toolset",
    host=MCP_HOST,
    port=MCP_PORT,
    streamable_http_path="/mcp",
)


def _safe_path(relative_path: str | None = ".") -> Path:
    """Resolve a user path under DATA_ROOT and block path traversal."""
    relative_path = relative_path or "."
    candidate = (DATA_ROOT / relative_path).resolve()
    data_root = DATA_ROOT.resolve()

    if candidate != data_root and data_root not in candidate.parents:
        raise ValueError(f"Path escapes DATA_ROOT: {relative_path}")

    return candidate


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "relative_path": str(path.relative_to(DATA_ROOT)),
        "type": "dir" if path.is_dir() else "file",
        "size_bytes": stat.st_size,
        "modified_epoch": int(stat.st_mtime),
    }


@mcp.tool()
def list_data_files(subdir: str = ".", pattern: str = "*", max_results: int = 100) -> list[dict[str, Any]]:
    """List files and folders inside the mounted shared data directory.

    Args:
        subdir: Subfolder under DATA_ROOT. Use "." for the root.
        pattern: Glob pattern, for example "*.csv" or "**/*.txt".
        max_results: Maximum number of entries to return.
    """
    root = _safe_path(subdir)
    if not root.exists():
        return [{"error": f"Subdir does not exist: {subdir}"}]
    if not root.is_dir():
        return [{"error": f"Not a directory: {subdir}"}]

    entries = []
    for path in sorted(root.glob(pattern)):
        try:
            entries.append(_file_info(path))
        except Exception as exc:  # Defensive: do not break the whole tool call.
            entries.append({"path": str(path), "error": str(exc)})
        if len(entries) >= max_results:
            break
    return entries


@mcp.tool()
def read_text_file(path: str, max_chars: int = 20000) -> str:
    """Read a text-like file from the shared data directory.

    Args:
        path: Relative file path under DATA_ROOT.
        max_chars: Maximum characters to return.
    """
    file_path = _safe_path(path)
    if not file_path.exists():
        return f"File does not exist: {path}"
    if not file_path.is_file():
        return f"Not a file: {path}"

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        content = f.read(max_chars + 1)

    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n[TRUNCATED after {max_chars} characters]"
    return content


@mcp.tool()
def write_text_file(path: str, content: str, overwrite: bool = False) -> str:
    """Write a text file into the shared data directory. Disabled unless ALLOW_WRITE_TO_DATA=true.

    Args:
        path: Relative file path under DATA_ROOT.
        content: Text content to write.
        overwrite: Whether to overwrite an existing file.
    """
    if not ALLOW_WRITE_TO_DATA:
        return "Writing is disabled. Set ALLOW_WRITE_TO_DATA=true to enable this tool."

    file_path = _safe_path(path)
    if file_path.exists() and not overwrite:
        return f"File already exists and overwrite=false: {path}"

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


@mcp.tool()
def summarize_csv(path: str, max_rows: int = 5) -> dict[str, Any]:
    """Summarize a CSV file in the shared data directory.

    Args:
        path: Relative CSV path under DATA_ROOT.
        max_rows: Number of preview rows to return.
    """
    file_path = _safe_path(path)
    if not file_path.exists():
        return {"error": f"File does not exist: {path}"}
    if not file_path.is_file():
        return {"error": f"Not a file: {path}"}

    df = pd.read_csv(file_path)
    return {
        "path": path,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
        "preview": df.head(max_rows).to_dict(orient="records"),
        "numeric_summary": df.describe(include="number").fillna("").to_dict(),
    }


@mcp.tool()
def search_text_files(query: str, subdir: str = ".", glob_pattern: str = "**/*", max_matches: int = 20) -> list[dict[str, Any]]:
    """Search for text inside files in the shared data directory.

    Args:
        query: Case-insensitive text to search for.
        subdir: Subfolder under DATA_ROOT.
        glob_pattern: File glob pattern, for example "**/*.md".
        max_matches: Maximum matching lines to return.
    """
    root = _safe_path(subdir)
    if not root.exists() or not root.is_dir():
        return [{"error": f"Invalid directory: {subdir}"}]

    query_lower = query.lower()
    matches: list[dict[str, Any]] = []

    for file_path in sorted(root.glob(glob_pattern)):
        if not file_path.is_file():
            continue

        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    if query_lower in line.lower():
                        matches.append({
                            "relative_path": str(file_path.relative_to(DATA_ROOT)),
                            "line": line_no,
                            "text": line.strip()[:500],
                        })
                        if len(matches) >= max_matches:
                            return matches
        except Exception:
            # Binary or unreadable file; skip.
            continue

    return matches


@mcp.tool()
def get_data_root_info() -> dict[str, Any]:
    """Return basic information about the mounted shared data directory."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    entries = list(DATA_ROOT.iterdir())
    return {
        "data_root": str(DATA_ROOT),
        "exists": DATA_ROOT.exists(),
        "allow_write": ALLOW_WRITE_TO_DATA,
        "top_level_items": len(entries),
        "examples": [_file_info(p) for p in sorted(entries)[:10]],
    }


if __name__ == "__main__":
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Streamable HTTP exposes tools at http://host:port/mcp.
    # Host, port, and path are configured on the FastMCP instance above.
    mcp.run(transport="streamable-http")
