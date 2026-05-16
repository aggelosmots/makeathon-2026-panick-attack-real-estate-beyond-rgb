from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.common_config import env_int, env_str

### These environment variables can be set in the .env file or in the environment. 
### See .env.example for reference.
MODEL_PROVIDER  = env_str("MODEL_PROVIDER", "groq").strip().lower()
OLLAMA_HOST     = env_str("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL    = env_str("OLLAMA_MODEL", "qwen3:8b")
GROQ_API_BASE   = env_str("GROQ_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_API_KEY    = env_str("GROQ_API_KEY", "")
GROQ_MODEL      = env_str("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_MAX_COMPLETION_TOKENS = env_int("GROQ_MAX_COMPLETION_TOKENS", 512)
MCP_SERVER_URL  = env_str("MCP_SERVER_URL", "http://localhost:8000/mcp")
AGENT_MAX_STEPS = env_int("AGENT_MAX_STEPS", 6)

###
###
###
###
###         WE NEED TO CREATE THE AGENT PROMPT
SYSTEM_PROMPT = """You are a local agent running inside Docker.
You can use MCP tools provided to you to access and perform actions on the system and data.
"""
###
###
###

class OllamaModelNotFoundError(RuntimeError):
    """Raised when the selected Ollama model is not available locally."""


class ModelAPIError(RuntimeError):
    """Raised when a model provider request fails with telemetry attached."""

    def __init__(self, message: str, telemetry: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.telemetry = telemetry or {}


def _obj_to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def default_model_for_provider(provider: str | None = None) -> str:
    provider = (provider or MODEL_PROVIDER).strip().lower()
    if provider == "groq":
        return GROQ_MODEL
    if provider == "ollama":
        return OLLAMA_MODEL
    return GROQ_MODEL


def _mcp_tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    tool_dict = _obj_to_dict(tool)
    name = tool_dict.get("name", "")
    description = (tool_dict.get("description") or f"MCP tool: {name}").split("\n\n")[0].strip()
    parameters = (
        tool_dict.get("inputSchema")
        or tool_dict.get("input_schema")
        or {"type": "object", "properties": {}}
    )

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _tool_result_to_text(result: Any) -> str:
    result_dict = _obj_to_dict(result)

    if isinstance(result_dict, dict) and "content" in result_dict:
        parts = []
        for item in result_dict["content"]:
            item_dict = _obj_to_dict(item)
            if isinstance(item_dict, dict):
                if item_dict.get("type") == "text":
                    parts.append(str(item_dict.get("text", "")))
                else:
                    parts.append(json.dumps(item_dict, ensure_ascii=False))
            else:
                text = getattr(item, "text", None)
                parts.append(str(text if text is not None else item))
        return "\n".join(p for p in parts if p)

    return json.dumps(result_dict, ensure_ascii=False, default=str)


def _exception_summary(exc: BaseException) -> str:
    sub_exceptions = getattr(exc, "exceptions", None)
    if sub_exceptions:
        return "; ".join(_exception_summary(sub_exc) for sub_exc in sub_exceptions)
    return f"{type(exc).__name__}: {exc}"


def _api_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()

    if isinstance(payload, dict) and payload.get("error"):
        error = payload["error"]
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        return str(error)
    return response.text.strip()


def _rate_limit_headers(response: httpx.Response) -> dict[str, str]:
    header_names = [
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    ]
    return {name: response.headers[name] for name in header_names if name in response.headers}


def _parse_rate_limit_message(message: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}

    for key in ("Limit", "Used", "Requested"):
        match = re.search(rf"{key}\s+(\d+)", message, flags=re.IGNORECASE)
        if match:
            parsed[key.lower()] = int(match.group(1))

    retry_match = re.search(r"try again in\s+([\d.]+)\s*([a-z]+)", message, flags=re.IGNORECASE)
    if retry_match:
        parsed["retry_after"] = f"{retry_match.group(1)}{retry_match.group(2)}"

    return parsed


def _response_telemetry(
    provider: str,
    model: str,
    response: httpx.Response,
    payload: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "status_code": response.status_code,
        "rate_limit_headers": _rate_limit_headers(response),
    }

    request_id = response.headers.get("x-request-id")
    if request_id:
        telemetry["request_id"] = request_id

    if payload and payload.get("usage"):
        telemetry["usage"] = payload["usage"]

    if error_message:
        telemetry["error"] = error_message
        rate_limit = _parse_rate_limit_message(error_message)
        if rate_limit:
            telemetry["rate_limit_error"] = rate_limit

    return telemetry


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def list_ollama_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        response.raise_for_status()
        payload = response.json()

    models = payload.get("models") or []
    return sorted(str(model.get("name", "")) for model in models if model.get("name"))


async def _ensure_ollama_model(client: httpx.AsyncClient, model: str) -> None:
    response = await client.get(f"{OLLAMA_HOST}/api/tags")
    response.raise_for_status()
    payload = response.json()

    available_models = sorted(
        str(item.get("name", "")) for item in payload.get("models", []) if item.get("name")
    )
    if model in available_models:
        return

    installed = ", ".join(available_models) if available_models else "none"
    raise OllamaModelNotFoundError(
        f"Ollama model `{model}` is not installed. Installed models: {installed}. "
        f"Pull it with `docker compose exec ollama ollama pull {model}`."
    )


async def list_groq_models() -> list[str]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{GROQ_API_BASE}/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        )
        response.raise_for_status()
        payload = response.json()

    models = payload.get("data") or []
    return sorted(str(model.get("id", "")) for model in models if model.get("id"))


async def list_provider_models(provider: str | None = None) -> list[str]:
    provider = (provider or MODEL_PROVIDER).strip().lower()
    if provider == "groq":
        return await list_groq_models()
    if provider == "ollama":
        return await list_ollama_models()
    raise ValueError(f"Unsupported model provider: {provider}")


async def list_mcp_tools() -> list[dict[str, Any]]:
    try:
        async with streamable_http_client(MCP_SERVER_URL) as stream_tuple:
            read_stream, write_stream = stream_tuple[0], stream_tuple[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                return [_obj_to_dict(t) for t in listed.tools]
    except Exception as exc:
        raise RuntimeError(f"Could not list MCP tools from {MCP_SERVER_URL}: {_exception_summary(exc)}") from exc


async def _call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    async with streamable_http_client(MCP_SERVER_URL) as stream_tuple:
        read_stream, write_stream = stream_tuple[0], stream_tuple[1]
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments or {})
            return _tool_result_to_text(result)


async def _get_model_tools() -> list[dict[str, Any]]:
    mcp_tools = await list_mcp_tools()
    return [_mcp_tool_to_openai_schema(t) for t in mcp_tools]


def _base_messages(user_text: str, history: list[dict[str, str]] | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history:
            if msg.get("role") in {"user", "assistant"}:
                messages.append({"role": msg["role"], "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_text})
    return messages


async def _run_ollama_agent(
    user_text: str,
    history: list[dict[str, str]] | None,
    model: str,
    max_steps: int,
) -> dict[str, Any]:
    messages = _base_messages(user_text, history)
    tools = await _get_model_tools()
    trace: list[dict[str, Any]] = []
    telemetry: dict[str, Any] = {
        "provider": "ollama",
        "model": model,
        "calls": [],
    }

    async with httpx.AsyncClient(timeout=None) as client:
        await _ensure_ollama_model(client, model)

        for _ in range(max_steps):
            response = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                },
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = _api_error_message(response)
                call_telemetry = _response_telemetry("ollama", model, response, error_message=detail)
                telemetry["calls"].append(call_telemetry)
                raise ModelAPIError(
                    f"Ollama chat request failed for model `{model}`: {detail}",
                    telemetry,
                ) from exc

            payload = response.json()
            telemetry["calls"].append(_response_telemetry("ollama", model, response, payload))
            assistant_message = payload.get("message", {})
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                return {
                    "answer": assistant_message.get("content", ""),
                    "trace": trace,
                    "messages": messages,
                    "telemetry": telemetry,
                }

            for call in tool_calls:
                fn = call.get("function", {})
                tool_name = fn.get("name")
                arguments = _parse_tool_arguments(fn.get("arguments") or {})

                if not tool_name:
                    continue

                trace.append({"tool": tool_name, "arguments": arguments})
                try:
                    tool_result = await _call_mcp_tool(tool_name, arguments)
                except Exception as exc:
                    tool_result = f"Tool error: {type(exc).__name__}: {exc}"

                trace[-1]["result_preview"] = tool_result[:1000]
                messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": tool_result,
                })

    return {
        "answer": "I reached the configured tool-call limit before producing a final answer.",
        "trace": trace,
        "messages": messages,
        "telemetry": telemetry,
    }


async def _run_groq_agent(
    user_text: str,
    history: list[dict[str, str]] | None,
    model: str,
    max_steps: int,
) -> dict[str, Any]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Create one in GroqCloud and add it to `.env`.")

    messages = _base_messages(user_text, history)
    tools = await _get_model_tools()
    trace: list[dict[str, Any]] = []
    telemetry: dict[str, Any] = {
        "provider": "groq",
        "model": model,
        "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
        "calls": [],
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=None) as client:
        for _ in range(max_steps):
            response = await client.post(
                f"{GROQ_API_BASE}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": False,
                    "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
                },
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = _api_error_message(response)
                call_telemetry = _response_telemetry("groq", model, response, error_message=detail)
                telemetry["calls"].append(call_telemetry)
                raise ModelAPIError(
                    f"Groq chat request failed for model `{model}`: {detail}",
                    telemetry,
                ) from exc

            payload = response.json()
            telemetry["calls"].append(_response_telemetry("groq", model, response, payload))
            choices = payload.get("choices") or []
            assistant_message = (choices[0].get("message") if choices else {}) or {}
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                return {
                    "answer": assistant_message.get("content", ""),
                    "trace": trace,
                    "messages": messages,
                    "telemetry": telemetry,
                }

            for call in tool_calls:
                fn = call.get("function", {})
                tool_name = fn.get("name")
                arguments = _parse_tool_arguments(fn.get("arguments"))

                if not tool_name:
                    continue

                trace.append({"tool": tool_name, "arguments": arguments})
                try:
                    tool_result = await _call_mcp_tool(tool_name, arguments)
                except Exception as exc:
                    tool_result = f"Tool error: {type(exc).__name__}: {exc}"

                trace[-1]["result_preview"] = tool_result[:1000]
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": tool_name,
                    "content": tool_result,
                })

    return {
        "answer": "I reached the configured tool-call limit before producing a final answer.",
        "trace": trace,
        "messages": messages,
        "telemetry": telemetry,
    }


async def ask_agent(
    user_text: str,
    history: list[dict[str, str]] | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run a model + MCP tool-calling loop and return final answer plus trace."""
    provider = (provider or MODEL_PROVIDER).strip().lower()
    model = model or default_model_for_provider(provider)
    max_steps = max_steps or AGENT_MAX_STEPS

    if provider == "groq":
        return await _run_groq_agent(user_text, history, model, max_steps)
    if provider == "ollama":
        return await _run_ollama_agent(user_text, history, model, max_steps)
    raise ValueError(f"Unsupported model provider: {provider}")


def ask_agent_sync(
    user_text: str,
    history: list[dict[str, str]] | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        ask_agent(user_text, history=history, provider=provider, model=model, max_steps=max_steps)
    )
