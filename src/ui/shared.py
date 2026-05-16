from __future__ import annotations

import asyncio
import html
from typing import Any

import streamlit as st

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

MAX_STEPS = env_int("AGENT_MAX_STEPS", 6)
PROVIDERS = ["huggingface"]
USER_ROUTE = "/"
DEVELOPER_ROUTE = "/devel"


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def init_session_state() -> None:
    default_provider = MODEL_PROVIDER if MODEL_PROVIDER in PROVIDERS else PROVIDERS[0]
    defaults = {
        "provider": default_provider,
        "model": default_model_for_provider(default_provider),
        "max_steps": MAX_STEPS,
        "system_prompt": SYSTEM_PROMPT,
        "mcp_tools": [],
        "provider_models": [],
        "chat_history": [],
        "trace_history": [],
        "telemetry_history": [],
        "latest_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_runtime_settings() -> None:
    default_provider = MODEL_PROVIDER if MODEL_PROVIDER in PROVIDERS else PROVIDERS[0]
    st.session_state.provider = default_provider
    st.session_state.model = default_model_for_provider(default_provider)
    st.session_state.max_steps = MAX_STEPS
    st.session_state.system_prompt = SYSTEM_PROMPT


def clear_conversation_state() -> None:
    st.session_state.chat_history = []
    st.session_state.trace_history = []
    st.session_state.latest_result = None


def clear_telemetry_state() -> None:
    st.session_state.telemetry_history = []
    if st.session_state.latest_result:
        st.session_state.latest_result["telemetry"] = None


def clear_tool_cache() -> None:
    st.session_state.mcp_tools = []
    st.session_state.provider_models = []


def _short_description(description: str | None) -> str:
    if not description:
        return "No description provided."
    return description.split("\n\n")[0].strip()


def _schema_type(schema: dict[str, Any] | None) -> str:
    if not schema:
        return "any"
    schema_type = schema.get("type", "object")
    if schema_type == "array":
        items = schema.get("items", {})
        return f"array<{items.get('type', 'item')}>"
    return str(schema_type)


def _output_label(tool: dict[str, Any]) -> str:
    output_schema = tool.get("outputSchema") or {}
    result_schema = (output_schema.get("properties") or {}).get("result")
    return _schema_type(result_schema or output_schema)


def _latest_call(telemetry: dict[str, Any]) -> dict[str, Any]:
    calls = telemetry.get("calls") or []
    return calls[-1] if calls else telemetry


def _metric_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _last_telemetry() -> dict[str, Any] | None:
    for telemetry in reversed(st.session_state.telemetry_history):
        if telemetry:
            return telemetry
    return None


def latest_result() -> dict[str, Any] | None:
    return st.session_state.latest_result


def refresh_provider_models() -> list[str]:
    st.session_state.provider_models = run_async(list_provider_models(st.session_state.provider))
    return st.session_state.provider_models


def refresh_mcp_tools() -> list[dict[str, Any]]:
    st.session_state.mcp_tools = run_async(list_mcp_tools())
    return st.session_state.mcp_tools


def run_agent_turn(prompt: str) -> dict[str, Any]:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    try:
        result = run_async(
            ask_agent(
                prompt,
                history=st.session_state.chat_history[:-1],
                provider=st.session_state.provider,
                model=st.session_state.model,
                max_steps=int(st.session_state.max_steps),
                system_prompt=st.session_state.system_prompt,
            )
        )
        answer = result["answer"] or "_No text response returned._"
        trace = result.get("trace", [])
        telemetry = result.get("telemetry")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "telemetry": telemetry,
        })
        st.session_state.trace_history.append(trace)
        st.session_state.telemetry_history.append(telemetry)
        st.session_state.latest_result = {
            "content": answer,
            "trace": trace,
            "telemetry": telemetry,
            "is_error": False,
        }
        return st.session_state.latest_result
    except Exception as exc:
        error = f"Agent error: `{type(exc).__name__}: {exc}`"
        telemetry = getattr(exc, "telemetry", None)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error,
            "telemetry": telemetry,
        })
        st.session_state.telemetry_history.append(telemetry)
        st.session_state.latest_result = {
            "content": error,
            "trace": [],
            "telemetry": telemetry,
            "is_error": True,
        }
        return st.session_state.latest_result


def render_model_telemetry_details(telemetry: dict[str, Any] | None) -> None:
    if not telemetry:
        return
    latest = _latest_call(telemetry)
    usage = latest.get("usage") or {}
    rate_headers = latest.get("rate_limit_headers") or {}
    rate_error = latest.get("rate_limit_error") or {}

    top_cols = st.columns(4)
    top_cols[0].metric("Provider", _metric_value(telemetry.get("provider")))
    top_cols[1].metric("Model", _metric_value(telemetry.get("model")))
    top_cols[2].metric("HTTP", _metric_value(latest.get("status_code")))
    top_cols[3].metric("Calls", _metric_value(len(telemetry.get("calls") or [])))

    if telemetry.get("max_completion_tokens") is not None:
        st.caption(f"Max completion tokens: `{telemetry['max_completion_tokens']}`")

    if usage:
        st.markdown("**Token Usage**")
        usage_cols = st.columns(4)
        usage_cols[0].metric("Prompt", _metric_value(usage.get("prompt_tokens")))
        usage_cols[1].metric("Completion", _metric_value(usage.get("completion_tokens")))
        usage_cols[2].metric("Total", _metric_value(usage.get("total_tokens")))
        usage_cols[3].metric("Queue time", _metric_value(usage.get("queue_time")))

    if rate_headers:
        st.markdown("**Rate Limit Headers**")
        header_cols = st.columns(3)
        header_cols[0].metric("Token limit", _metric_value(rate_headers.get("x-ratelimit-limit-tokens")))
        header_cols[1].metric("Tokens remaining", _metric_value(rate_headers.get("x-ratelimit-remaining-tokens")))
        header_cols[2].metric("Token reset", _metric_value(rate_headers.get("x-ratelimit-reset-tokens")))

        request_cols = st.columns(3)
        request_cols[0].metric("Request limit", _metric_value(rate_headers.get("x-ratelimit-limit-requests")))
        request_cols[1].metric(
            "Requests remaining",
            _metric_value(rate_headers.get("x-ratelimit-remaining-requests")),
        )
        request_cols[2].metric("Retry after", _metric_value(rate_headers.get("retry-after")))

    if rate_error:
        st.markdown("**Parsed Rate Limit Error**")
        error_cols = st.columns(4)
        error_cols[0].metric("Limit", _metric_value(rate_error.get("limit")))
        error_cols[1].metric("Used", _metric_value(rate_error.get("used")))
        error_cols[2].metric("Requested", _metric_value(rate_error.get("requested")))
        error_cols[3].metric("Retry after", _metric_value(rate_error.get("retry_after")))

    if latest.get("error"):
        st.error(latest["error"])

    with st.expander("Raw telemetry"):
        st.json(telemetry)


def render_model_telemetry(telemetry: dict[str, Any] | None, expanded: bool = False) -> None:
    if not telemetry:
        return
    with st.expander("Model telemetry", expanded=expanded):
        render_model_telemetry_details(telemetry)


def render_telemetry_panel() -> None:
    st.subheader("Telemetry")
    telemetry = _last_telemetry()
    if not telemetry:
        st.caption("No model telemetry yet.")
        return

    latest = _latest_call(telemetry)
    usage = latest.get("usage") or {}
    rate_headers = latest.get("rate_limit_headers") or {}
    rate_error = latest.get("rate_limit_error") or {}

    st.metric("Provider", _metric_value(telemetry.get("provider")))
    st.caption(f"Model: `{_metric_value(telemetry.get('model'))}`")
    st.metric("HTTP", _metric_value(latest.get("status_code")))

    if usage:
        col_a, col_b = st.columns(2)
        col_a.metric("Prompt", _metric_value(usage.get("prompt_tokens")))
        col_b.metric("Total", _metric_value(usage.get("total_tokens")))

    if rate_headers:
        col_a, col_b = st.columns(2)
        col_a.metric("Token limit", _metric_value(rate_headers.get("x-ratelimit-limit-tokens")))
        col_b.metric("Remaining", _metric_value(rate_headers.get("x-ratelimit-remaining-tokens")))
        st.caption(f"Token reset: `{_metric_value(rate_headers.get('x-ratelimit-reset-tokens'))}`")

    if rate_error:
        col_a, col_b = st.columns(2)
        col_a.metric("Used", _metric_value(rate_error.get("used")))
        col_b.metric("Requested", _metric_value(rate_error.get("requested")))
        st.metric("Limit", _metric_value(rate_error.get("limit")))
        st.caption(f"Retry after: `{_metric_value(rate_error.get('retry_after'))}`")

    if latest.get("error"):
        st.error(latest["error"])


def render_tool_catalog(tools: list[dict[str, Any]]) -> None:
    st.subheader("MCP Tools")
    st.caption(f"{len(tools)} tools exposed by the MCP server")
    st.markdown(
        """
        <style>
            .tool-name {
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
                overflow-wrap: anywhere;
            }
            .tool-description {
                min-height: 3.2rem;
                color: rgba(49, 51, 63, 0.76);
                line-height: 1.35;
                margin-bottom: 0.75rem;
                overflow-wrap: anywhere;
            }
            .tool-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem;
                margin: 0.35rem 0 0.75rem;
            }
            .tool-badge {
                border: 1px solid rgba(49, 51, 63, 0.18);
                border-radius: 999px;
                padding: 0.16rem 0.45rem;
                font-size: 0.75rem;
                line-height: 1.35;
                background: rgba(49, 51, 63, 0.04);
                overflow-wrap: anywhere;
            }
            .tool-badge-required {
                border-color: rgba(214, 90, 49, 0.38);
                background: rgba(214, 90, 49, 0.09);
            }
            .tool-output {
                font-size: 0.78rem;
                color: rgba(49, 51, 63, 0.68);
                margin-top: 0.4rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2)
    for index, tool in enumerate(tools):
        with columns[index % 2]:
            with st.container(border=True):
                name = html.escape(str(tool.get("name", "unnamed_tool")))
                description = html.escape(_short_description(tool.get("description")))
                input_schema = tool.get("inputSchema") or {}
                properties = input_schema.get("properties") or {}
                required = set(input_schema.get("required") or [])

                st.markdown(f'<div class="tool-name">{name}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tool-description">{description}</div>', unsafe_allow_html=True)

                if properties:
                    badges = []
                    for arg_name, arg_schema in properties.items():
                        arg_type = _schema_type(arg_schema)
                        required_class = " tool-badge-required" if arg_name in required else ""
                        label = f"{arg_name}: {arg_type}"
                        if arg_name in required:
                            label += " *"
                        badges.append(
                            f'<span class="tool-badge{required_class}">{html.escape(label)}</span>'
                        )
                    st.markdown(
                        f'<div class="tool-badges">{"".join(badges)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="tool-badges"><span class="tool-badge">no arguments</span></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="tool-output">Output: {html.escape(_output_label(tool))}</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("Schema"):
                    st.json(tool)


def render_runtime_summary() -> None:
    st.metric("Provider", st.session_state.provider)
    st.metric("Model", st.session_state.model)
    st.metric("Max steps", int(st.session_state.max_steps))
    st.caption(f"Shared data path: `{DATA_ROOT}`")
    st.caption(f"MCP server: `{env_str('MCP_SERVER_URL', 'http://localhost:8000/mcp')}`")
    st.caption(f"Hugging Face API: `{HF_API_BASE}`")
