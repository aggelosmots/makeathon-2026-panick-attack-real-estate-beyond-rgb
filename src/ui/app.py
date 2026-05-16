from __future__ import annotations

import asyncio
import html
from typing import Any

import streamlit as st

from src.agent.agent import (
    GROQ_API_BASE,
    MODEL_PROVIDER,
    OLLAMA_HOST,
    ask_agent,
    default_model_for_provider,
    list_mcp_tools,
    list_provider_models,
)
from src.common_config import DATA_ROOT, env_int, env_str

MAX_STEPS = env_int("AGENT_MAX_STEPS", 6)
PROVIDERS = ["groq", "ollama"]


def run_async(coro):
    """Run an async function from Streamlit's synchronous execution model."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Streamlit normally has no running loop here, but keep this for portability.
    return loop.run_until_complete(coro)


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


def _render_tool_catalog(tools: list[dict[str, Any]]) -> None:
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


st.set_page_config(page_title="MCP Agent Template", layout="wide")

if "mcp_tools" not in st.session_state:
    st.session_state.mcp_tools = []

if "show_mcp_tools" not in st.session_state:
    st.session_state.show_mcp_tools = False

st.title("Welcome to Real Estate Beyond RGB")

with st.sidebar:
    st.header("Runtime")
    provider_index = PROVIDERS.index(MODEL_PROVIDER) if MODEL_PROVIDER in PROVIDERS else 0
    provider = st.selectbox("Model provider", options=PROVIDERS, index=provider_index)
    model = st.text_input("Model", value=default_model_for_provider(provider))
    max_steps = st.number_input("Max tool-call steps", min_value=1, max_value=20, value=MAX_STEPS)
    st.caption(f"Shared data path inside container: `{DATA_ROOT}`")
    st.caption(f"MCP server: `{env_str('MCP_SERVER_URL', 'http://localhost:8000/mcp')}`")
    if provider == "groq":
        st.caption(f"Groq API: `{GROQ_API_BASE}`")
    else:
        st.caption(f"Ollama host: `{OLLAMA_HOST}`")

    if st.button("Show provider models", use_container_width=True):
        try:
            available_models = run_async(list_provider_models(provider))
            if available_models:
                st.write(available_models)
            else:
                st.info("No models were returned by this provider.")
        except Exception as exc:
            st.error(f"Could not list provider models: {exc}")

    if st.button("Show MCP tools", use_container_width=True):
        try:
            st.session_state.mcp_tools = run_async(list_mcp_tools())
            st.session_state.show_mcp_tools = True
        except Exception as exc:
            st.error(f"Could not list MCP tools: {exc}")

    if st.session_state.show_mcp_tools and st.button("Hide MCP tools", use_container_width=True):
        st.session_state.show_mcp_tools = False


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "trace_history" not in st.session_state:
    st.session_state.trace_history = []

if st.session_state.show_mcp_tools:
    _render_tool_catalog(st.session_state.mcp_tools)
    st.divider()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask the agent about your mounted data...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running local agent..."):
            try:
                result = run_async(
                    ask_agent(
                        prompt,
                        history=st.session_state.chat_history[:-1],
                        provider=provider,
                        model=model,
                        max_steps=int(max_steps),
                    )
                )
                answer = result["answer"] or "_No text response returned._"
                st.markdown(answer)

                trace = result.get("trace", [])
                if trace:
                    with st.expander("Tool trace"):
                        st.json(trace)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.trace_history.append(trace)
            except Exception as exc:
                error = f"Agent error: `{type(exc).__name__}: {exc}`"
                st.error(error)
                st.session_state.chat_history.append({"role": "assistant", "content": error})
