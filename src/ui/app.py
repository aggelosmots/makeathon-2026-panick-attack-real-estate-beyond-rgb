from __future__ import annotations

import streamlit as st

from src.ui.shared import init_session_state, latest_result, run_agent_turn

st.set_page_config(
    page_title="Real Estate Beyond RGB",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_session_state()

st.title("Real Estate Beyond RGB")
st.caption("User workspace")

st.markdown(
    """
    <style>
        .panel-copy {
            color: rgba(49, 51, 63, 0.76);
            margin-bottom: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

results_col, chat_col = st.columns([1.75, 1], gap="large")

with results_col:
    st.subheader("Results")
    st.markdown(
        '<div class="panel-copy">Plots, structured outputs, and the latest agent response appear here.</div>',
        unsafe_allow_html=True,
    )

    result = latest_result()
    with st.container(border=True):
        if not result:
            st.info("No results yet. Submit a prompt from the panel on the right.")
        else:
            if result.get("is_error"):
                st.error(result["content"])
            else:
                st.markdown(result["content"])

            trace = result.get("trace") or []
            if trace:
                with st.expander("Tool activity", expanded=True):
                    for index, step in enumerate(trace, start=1):
                        with st.container(border=True):
                            st.markdown(f"**Step {index}**")
                            st.caption(f"Tool: `{step.get('tool', 'unknown')}`")
                            if step.get("arguments"):
                                st.json(step["arguments"])
                            preview = step.get("result_preview")
                            if preview:
                                st.code(preview, language="text")

with chat_col:
    st.subheader("Talk to the Agent")
    st.markdown(
        '<div class="panel-copy">Use the prompt panel to send requests to the agent.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        with st.form("user_prompt_form", clear_on_submit=True):
            prompt = st.text_area(
                "Prompt",
                height=180,
                label_visibility="collapsed",
                placeholder="Ask the agent about your data, files, or workflow...",
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted:
            if not prompt.strip():
                st.warning("Enter a prompt before sending it.")
            else:
                with st.spinner("Running agent..."):
                    run_agent_turn(prompt.strip())
                st.rerun()

    if st.session_state.chat_history:
        st.subheader("Conversation")
        for message in st.session_state.chat_history[-8:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
