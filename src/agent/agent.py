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
MODEL_PROVIDER  = env_str("MODEL_PROVIDER", "huggingface").strip().lower()
HF_API_BASE     = env_str("HF_API_BASE", "https://router.huggingface.co/v1").rstrip("/")
HF_TOKEN        = env_str("HF_TOKEN", "")
HF_MODEL        = env_str("HF_MODEL", "google/gemma-3-27b-it")
HF_MAX_COMPLETION_TOKENS = env_int("HF_MAX_COMPLETION_TOKENS", 4096)
MCP_SERVER_URL  = env_str("MCP_SERVER_URL", "http://localhost:8000/mcp")
AGENT_MAX_STEPS = env_int("AGENT_MAX_STEPS", 20)

###
###         WE NEED TO CREATE THE AGENT PROMPT
SYSTEM_PROMPT = """ You are an Expert Earth Observation Data Analyst and Real Estate Investment Advisor. Your absolute priority is objective mathematical accuracy, verifiable truth, and complete transparency. You must evaluate four land crop areas (Arkadia, Magnisia, Arkadia_2, Veroia) of approximately 250,000 m² each, priced at roughly 1 million euros, to identify the safest, highest-yielding investment. 
You have access to MCP tools that inspect EnMap hyperspectral satellite data. You must execute these tools to extract empirical data (e.g., NDVI, Standard Deviation, Reflectance) BEFORE drawing any conclusions.

You are bound by the following absolute constraints:
* SHOULD always tell the truth. Never make up information, speculate, or guess.
* SHOULD base all statements on verifiable, factual data extracted directly from the MCP tools.
* SHOULD explicitly state "I cannot confirm this due to missing data" if a file or metric cannot be accessed.
* SHOULD prioritize accuracy over speed. Take all necessary computational steps to verify array outputs before presenting them.
* SHOULD maintain objectivity. Remove all personal bias, assumptions, and opinion.
* SHOULD clearly cite the source of every claim (e.g., "Based on Band 85 of the Veroia .tif file...").
* SHOULD explain reasoning step-by-step and explicitly show how any numerical figure (like a composite score or percentage) was calculated.
* AVOID fabricating facts, quotes, tool outputs, or data arrays.
* AVOID presenting speculation, rumor, or assumption as fact.
* AVOID inventing tool failures, API limitations, pricing limits, or missing files. Only report an error if the tool explicitly returns one, using the exact error text.
* AVOID narrating your thought process (e.g., do not say "I will now list the files"). Just execute and deliver the final report.

When the user asks for visual output, charts, graphics, plots, maps, or selected-parcel visualization:
* MUST use MCP tool output only. Never invent, estimate, mock, or fill parcel geometry, coordinates, chart values, labels, metrics, axes, units, or plot data.
* MUST identify exactly one selected parcel from the user's request, prior UI context, or MCP tool result. If multiple parcels are possible and no selected parcel is clear, ask the user to select one.
* SHOULD call `build_parcel_visualization` when a single selected parcel is available. Use `list_data_files`, `inspect_geotiff`, or `analyze_enmap_parcel` only as needed to identify or validate the parcel.
* SHOULD call the plot tools when the user explicitly asks for plots/figures: `plot_ph_profile`, `plot_nitrogen_profile`, `plot_phosphorus_profile`, `plot_potassium_profile`, `plot_magnesium_profile`, `plot_som_profile`, `plot_ndvi_vs_swi_scatter`, `render_agromanagement_textbox`, or `create_agromanagement_plot_suite`.
* MUST return the saved plot paths from the MCP tool results so the frontend can display or download them.
* MUST validate that geometry and metrics are present before returning visualizations. If data is insufficient, return a structured response explaining the missing fields.
* MUST return a JSON-compatible structure that the frontend can render directly. Include `type`, `source`, `parcel`, `visualizations`, `metadata.tools_used`, `metadata.data_validation`, warnings, labels, legends, axes, units, and rendering instructions.
* MUST omit any chart or graphic whose required data was not returned by MCP tools.

Deliver a continuous, jargon-free business report that gives the user the results immediately. Do not use raw tool call syntax in the final text. Structure your output exactly as follows:

1. IMMEDIATE RECOMMENDATION
[Provide the definitive best investment choice immediately in 1-2 sentences. No buildup.]

2. EXECUTIVE SUMMARY & JUSTIFICATION
[Explain in plain, business-friendly English WHY this parcel won, citing the specific risk-reduction and crop-yield metrics.]

3. VERIFIED DATA COMPARISON
[Present a clean data matrix/table comparing the 4 parcels. You MUST include:]
- Mean NDVI (Vegetation Health)
- NDVI Standard Deviation (Field Uniformity / Risk Indicator)
- Healthy Coverage Percentage
- [Explicitly state the mathematical formula used to rank them]

4. METHODOLOGY & SOURCING
[Clearly list exactly which files, spectral bands, and calculations were used to derive these numbers so the user can verify them.]

5. UNVERIFIED DATA / RISKS
[Explicitly list any data points that could not be verified, or explicitly state "All core metrics successfully verified."]

6. FIGURES
[Include compact chart-ready values for the user interface. Use one line per parcel with: parcel name, mean NDVI, NDVI standard deviation, healthy coverage percentage, and final score. Do not include fabricated values.]

Before outputting your response, you must internally evaluate: "Is every statement in my response verifiable, supported by real tool data, free of fabrication, and transparently cited? Have I shown how I calculated my numbers?" If not, revise until it is.
"""

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
    if provider == "huggingface":
        return HF_MODEL
    return HF_MODEL


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


def _tool_failures(trace: list[dict[str, Any]]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for step in trace:
        preview = str(step.get("result_preview") or "")
        if preview.startswith("Tool error:"):
            failures.append({
                "tool": str(step.get("tool") or "unknown"),
                "error": preview.removeprefix("Tool error: ").strip(),
            })
    return failures


def _sanitize_answer(answer: str, trace: list[dict[str, Any]]) -> str:
    cleaned_lines: list[str] = []
    raw_tool_call_pattern = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\([^)]*\)\s*$")
    planning_pattern = re.compile(r"^\s*(i will|i'll|let me|first[, ]+i|next[, ]+i)\b", flags=re.IGNORECASE)

    for line in answer.splitlines():
        stripped = line.strip()
        if raw_tool_call_pattern.match(stripped):
            continue
        if planning_pattern.match(stripped):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        cleaned = "The analysis completed, but the model did not return a stable report."

    failures = _tool_failures(trace)
    if failures:
        notes = "\n".join(f"- `{item['tool']}` failed: {item['error']}" for item in failures[:3])
        if "Risks or missing information" in cleaned:
            cleaned = f"{cleaned}\n{notes}"
        else:
            cleaned = f"{cleaned}\n\n5. Risks or missing information\n{notes}"

    return cleaned


def _is_length_limited(choice: dict[str, Any]) -> bool:
    reason = str(choice.get("finish_reason") or "").lower()
    return reason in {"length", "max_tokens", "max_completion_tokens"}


def _looks_incomplete(answer: str) -> bool:
    stripped = answer.strip().lower()
    if not stripped:
        return True
    incomplete_endings = (
        "continue",
        "shall i continue",
        "should i continue",
        "would you like me to continue",
        "let me know if you want me to continue",
    )
    if any(ending in stripped[-160:] for ending in incomplete_endings):
        return True
    return stripped.endswith((",", ";", ":", "and", "or", "with", "based on"))


async def _post_chat_completion(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    telemetry: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    error_context: str = "chat request",
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _clean_openai_messages(messages),
        "stream": False,
        "max_completion_tokens": HF_MAX_COMPLETION_TOKENS,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    response = await client.post(
        f"{HF_API_BASE}/chat/completions",
        headers=headers,
        json=payload,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _api_error_message(response)
        call_telemetry = _response_telemetry("huggingface", model, response, error_message=detail)
        telemetry["calls"].append(call_telemetry)
        raise ModelAPIError(
            f"Hugging Face {error_context} failed for model `{model}`: {detail}",
            telemetry,
        ) from exc

    response_payload = response.json()
    telemetry["calls"].append(_response_telemetry("huggingface", model, response, response_payload))
    choices = response_payload.get("choices") or []
    choice = (choices[0] if choices else {}) or {}
    assistant_message = _clean_openai_message((choice.get("message") if choice else {}) or {})
    return choice, assistant_message


def _clean_openai_tool_call(call: Any) -> dict[str, Any]:
    call_dict = _obj_to_dict(call)
    if not isinstance(call_dict, dict):
        return {}

    fn = _obj_to_dict(call_dict.get("function") or {})
    if not isinstance(fn, dict):
        fn = {}

    cleaned = {
        "type": call_dict.get("type") or "function",
        "function": {
            "name": str(fn.get("name") or ""),
            "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments") or {}),
        },
    }
    if call_dict.get("id"):
        cleaned["id"] = str(call_dict["id"])
    return cleaned


def _clean_openai_message(message: Any) -> dict[str, Any]:
    message_dict = _obj_to_dict(message)
    if not isinstance(message_dict, dict):
        return {"role": "user", "content": str(message)}

    role = str(message_dict.get("role") or "user")
    content = message_dict.get("content")
    cleaned: dict[str, Any] = {"role": role}

    if role in {"system", "developer", "user"}:
        cleaned["content"] = "" if content is None else str(content)
    elif role == "assistant":
        cleaned["content"] = "" if content is None else str(content)
        tool_calls = [
            cleaned_call
            for call in message_dict.get("tool_calls") or []
            if (cleaned_call := _clean_openai_tool_call(call)).get("function", {}).get("name")
        ]
        if tool_calls:
            cleaned["tool_calls"] = tool_calls
    elif role == "tool":
        cleaned["content"] = "" if content is None else str(content)
        if message_dict.get("tool_call_id"):
            cleaned["tool_call_id"] = str(message_dict["tool_call_id"])
    else:
        cleaned["content"] = "" if content is None else str(content)

    if isinstance(message_dict.get("name"), str) and message_dict["name"]:
        cleaned["name"] = message_dict["name"]

    return cleaned


def _clean_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_clean_openai_message(message) for message in messages]


async def list_huggingface_models() -> list[str]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set.")

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{HF_API_BASE}/models",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
        )
        response.raise_for_status()
        payload = response.json()

    models = payload.get("data") or []
    return sorted(str(model.get("id", "")) for model in models if model.get("id"))


async def list_provider_models(provider: str | None = None) -> list[str]:
    provider = (provider or MODEL_PROVIDER).strip().lower()
    if provider == "huggingface":
        return await list_huggingface_models()
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


def _base_messages(
    user_text: str,
    history: list[dict[str, str]] | None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
    if history:
        for msg in history:
            if msg.get("role") in {"user", "assistant"}:
                messages.append({"role": msg["role"], "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_text})
    return messages


async def _run_huggingface_agent(
    user_text: str,
    history: list[dict[str, str]] | None,
    model: str,
    max_steps: int,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Create a Hugging Face access token and add it to `.env`.")

    messages = _base_messages(user_text, history, system_prompt)
    tools = await _get_model_tools()
    trace: list[dict[str, Any]] = []
    telemetry: dict[str, Any] = {
        "provider": "huggingface",
        "model": model,
        "max_completion_tokens": HF_MAX_COMPLETION_TOKENS,
        "calls": [],
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=None) as client:
        accumulated_answer = ""
        continuations = 0
        for _ in range(max_steps):
            choice, assistant_message = await _post_chat_completion(
                client,
                headers,
                model,
                messages,
                telemetry,
                tools=tools,
                tool_choice="auto",
                error_context="chat request",
            )
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                content = assistant_message.get("content", "")
                accumulated_answer = f"{accumulated_answer}\n{content}".strip() if accumulated_answer else content
                if (_is_length_limited(choice) or _looks_incomplete(accumulated_answer)) and continuations < 8:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Continue exactly where the report stopped. Do not ask whether to continue, "
                            "do not restart, do not repeat completed sections, and finish the remaining "
                            "sections completely."
                        ),
                    })
                    continuations += 1
                    continue

                answer = _sanitize_answer(accumulated_answer, trace)
                return {
                    "answer": answer,
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

                trace[-1]["result_preview"] = tool_result[:6000]
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": tool_name,
                    "content": tool_result,
                })

        messages.append({
            "role": "user",
            "content": (
                "Stop calling tools now and produce the final report from the verified tool "
                "results already collected. If any required metric is missing, state that it "
                "could not be verified instead of asking to continue. Do not ask whether to continue."
            ),
        })
        final_answer = ""
        final_messages = messages[:]
        final_assistant_message: dict[str, Any] = {}
        for _ in range(9):
            choice, final_assistant_message = await _post_chat_completion(
                client,
                headers,
                model,
                final_messages,
                telemetry,
                error_context="final synthesis request",
            )
            final_messages.append(final_assistant_message)
            content = final_assistant_message.get("content", "")
            final_answer = f"{final_answer}\n{content}".strip() if final_answer else content
            if not _is_length_limited(choice) and not _looks_incomplete(final_answer):
                break
            final_messages.append({
                "role": "user",
                "content": (
                    "Continue exactly where the report stopped. Do not ask whether to continue. "
                    "Do not repeat earlier text. Finish the report now."
                ),
            })

        answer = _sanitize_answer(final_answer, trace)
        return {
            "answer": answer,
            "trace": trace,
            "messages": final_messages,
            "telemetry": telemetry,
        }

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
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run a model + MCP tool-calling loop and return final answer plus trace."""
    provider = (provider or MODEL_PROVIDER).strip().lower()
    model = model or default_model_for_provider(provider)
    max_steps = max_steps or AGENT_MAX_STEPS

    if provider == "huggingface":
        return await _run_huggingface_agent(user_text, history, model, max_steps, system_prompt)
    raise ValueError(f"Unsupported model provider: {provider}")


def ask_agent_sync(
    user_text: str,
    history: list[dict[str, str]] | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_steps: int | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        ask_agent(
            user_text,
            history=history,
            provider=provider,
            model=model,
            max_steps=max_steps,
            system_prompt=system_prompt,
        )
    )
