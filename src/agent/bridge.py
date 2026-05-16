from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from src.agent.agent import (
    HF_API_BASE,
    MODEL_PROVIDER,
    SYSTEM_PROMPT,
    ask_agent,
    default_model_for_provider,
    list_mcp_tools,
    list_provider_models,
)
from src.common_config import DATA_ROOT, env_int, env_str

MAX_STEPS = env_int("AGENT_MAX_STEPS", 20)
PROVIDERS = ["huggingface"]


def _default_provider() -> str:
    return MODEL_PROVIDER if MODEL_PROVIDER in PROVIDERS else PROVIDERS[0]


def _latest_call(telemetry: dict[str, Any] | None) -> dict[str, Any]:
    if not telemetry:
        return {}
    calls = telemetry.get("calls") or []
    return calls[-1] if calls else telemetry


def _compact_telemetry(telemetry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not telemetry:
        return None

    latest = _latest_call(telemetry)
    return {
        "provider": telemetry.get("provider"),
        "model": telemetry.get("model"),
        "status_code": latest.get("status_code"),
        "calls": len(telemetry.get("calls") or []),
        "usage": latest.get("usage") or {},
        "rate_limit_headers": latest.get("rate_limit_headers") or {},
        "rate_limit_error": latest.get("rate_limit_error") or {},
        "max_completion_tokens": telemetry.get("max_completion_tokens"),
        "error": latest.get("error"),
        "raw": telemetry,
    }


def _runtime_defaults() -> dict[str, Any]:
    provider = _default_provider()
    return {
        "provider": provider,
        "model": default_model_for_provider(provider),
        "max_steps": MAX_STEPS,
        "system_prompt": SYSTEM_PROMPT,
        "data_root": str(DATA_ROOT),
        "mcp_server": env_str("MCP_SERVER_URL", "http://localhost:8000/mcp"),
        "hf_api_base": HF_API_BASE,
        "user_route": "/",
        "developer_route": "/devel",
    }


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {}


def _clean_history(history: Any) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    if not isinstance(history, list):
        return cleaned

    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        cleaned.append({"role": role, "content": str(item.get("content") or "")})
    return cleaned


async def _run_agent_command() -> dict[str, Any]:
    payload = _read_payload()
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    provider = str(payload.get("provider") or _default_provider()).strip().lower()
    model = str(payload.get("model") or default_model_for_provider(provider)).strip()
    max_steps = int(payload.get("max_steps") or MAX_STEPS)
    system_prompt = str(payload.get("system_prompt") or SYSTEM_PROMPT)
    history = _clean_history(payload.get("history"))

    result = await ask_agent(
        prompt,
        history=history,
        provider=provider,
        model=model,
        max_steps=max_steps,
        system_prompt=system_prompt,
    )
    answer = result["answer"] or "_No text response returned._"
    trace = result.get("trace", [])
    telemetry = result.get("telemetry")

    return {
        "assistant_message": {
            "role": "assistant",
            "content": answer,
            "telemetry": telemetry,
        },
        "latest_result": {
            "content": answer,
            "trace": trace,
            "telemetry": telemetry,
            "is_error": False,
        },
        "latest_telemetry": _compact_telemetry(telemetry),
    }


async def _main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "defaults"
    try:
        if command == "defaults":
            payload: Any = _runtime_defaults()
        elif command == "run-agent":
            payload = await _run_agent_command()
        elif command == "tools":
            payload = await list_mcp_tools()
        elif command == "models":
            request = _read_payload()
            provider = str(request.get("provider") or _default_provider()).strip().lower()
            payload = await list_provider_models(provider)
        else:
            raise ValueError(f"Unknown bridge command: {command}")

        print(json.dumps({"ok": True, "data": payload}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
