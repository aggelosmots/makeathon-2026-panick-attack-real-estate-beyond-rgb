from __future__ import annotations

import streamlit as st

from src.ui.shared import (
    DEVELOPER_ROUTE,
    USER_ROUTE,
    clear_conversation_state,
    clear_telemetry_state,
    clear_tool_cache,
    init_session_state,
    refresh_mcp_tools,
    refresh_provider_models,
    render_model_telemetry_details,
    render_runtime_summary,
    render_telemetry_panel,
    render_tool_catalog,
    reset_runtime_settings,
)

st.set_page_config(
    page_title="Developer Console",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_session_state()

st.title("Developer Console")
st.caption("Hidden runtime workspace for model settings, telemetry, tools, and system controls.")

overview_col, runtime_col = st.columns([1, 1.4], gap="large")

with overview_col:
    with st.container(border=True):
        st.subheader("User View")
        st.markdown(f"Public route: `{USER_ROUTE}`")
        st.markdown(f"Developer route: `{DEVELOPER_ROUTE}`")
        st.write("The user view contains only a prompt panel, conversation, and a results workspace.")
        st.write("Use this page to control the runtime, inspect telemetry, manage the system prompt, and inspect tools.")
        st.metric("Messages in conversation", len(st.session_state.chat_history))
        st.metric("Telemetry snapshots", len(st.session_state.telemetry_history))
        latest = st.session_state.latest_result
        if latest:
            st.caption("Latest result preview")
            st.code(latest.get("content", "")[:600], language="text")

with runtime_col:
    with st.container(border=True):
        st.subheader("Runtime Settings")
        st.text_input("Model", key="model")
        st.number_input("Max tool-call steps", min_value=1, max_value=20, key="max_steps")
        st.text_area("System prompt", key="system_prompt", height=220)
        st.caption("These settings apply to the active Streamlit session.")

tabs = st.tabs(["Telemetry", "MCP Tools", "System"])

with tabs[0]:
    render_telemetry_panel()
    latest_telemetry = next(
        (telemetry for telemetry in reversed(st.session_state.telemetry_history) if telemetry),
        None,
    )
    if latest_telemetry:
        with st.expander("Detailed telemetry", expanded=True):
            render_model_telemetry_details(latest_telemetry)

with tabs[1]:
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Refresh MCP tools", use_container_width=True):
            try:
                refresh_mcp_tools()
            except Exception as exc:
                st.error(f"Could not list MCP tools: {exc}")
    with action_cols[1]:
        if st.button("Show provider models", use_container_width=True):
            try:
                refresh_provider_models()
            except Exception as exc:
                st.error(f"Could not list provider models: {exc}")

    if st.session_state.provider_models:
        st.subheader("Available Provider Models")
        st.write(st.session_state.provider_models)

    if st.session_state.mcp_tools:
        render_tool_catalog(st.session_state.mcp_tools)
    else:
        st.info("Refresh the MCP tool catalog to inspect the tools exposed to the agent.")

with tabs[2]:
    left_col, right_col = st.columns([1.1, 1], gap="large")

    with left_col:
        with st.container(border=True):
            st.subheader("System Runtime")
            render_runtime_summary()

    with right_col:
        with st.container(border=True):
            st.subheader("System Management")
            if st.button("Clear conversation", use_container_width=True):
                clear_conversation_state()
                st.rerun()
            if st.button("Clear telemetry", use_container_width=True):
                clear_telemetry_state()
                st.rerun()
            if st.button("Clear tool cache", use_container_width=True):
                clear_tool_cache()
                st.rerun()
            if st.button("Reset developer settings", use_container_width=True):
                reset_runtime_settings()
                st.rerun()
