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
st.caption("Land investment analysis workspace")

st.markdown(
    """
    <style>
        .overview {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            background: rgba(49, 51, 63, 0.03);
        }
        .overview-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .overview-copy {
            color: rgba(49, 51, 63, 0.78);
            line-height: 1.5;
            margin-bottom: 0.5rem;
        }
        .panel-copy {
            color: rgba(49, 51, 63, 0.76);
            margin-bottom: 0.85rem;
        }
        .report-shell {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            min-height: 28rem;
            background: rgba(255, 255, 255, 0.02);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="overview">
        <div class="overview-title">Challenge Brief</div>
        <div class="overview-copy">
            Compare four EnMap hyperspectral parcel datasets and determine which land area is the strongest investment opportunity.
            Each area is about 250,000 m², has a similar footprint, and is assumed to cost around 1 million euros.
        </div>
        <div class="overview-copy">
            The report should use spectral and environmental indicators, explain the methodology, and present a transparent recommendation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("user_prompt_form", clear_on_submit=True):
    input_col, button_col = st.columns([6.5, 1], gap="small")
    with input_col:
        prompt = st.text_input(
            "Prompt",
            label_visibility="collapsed",
            placeholder="Compare the four parcel data files and recommend the best investment opportunity...",
        )
    with button_col:
        st.write("")
        submitted = st.form_submit_button("Send", use_container_width=True)

if submitted:
    if not prompt.strip():
        st.warning("Enter a request before sending it.")
    else:
        with st.spinner("Running agent analysis..."):
            run_agent_turn(prompt.strip())
        st.rerun()

st.subheader("Condensed Report")
st.markdown(
    '<div class="panel-copy">The agent analyzes the available parcel data and returns a compact comparison report here.</div>',
    unsafe_allow_html=True,
)

result = latest_result()
with st.container():
    st.markdown('<div class="report-shell">', unsafe_allow_html=True)
    if not result:
        st.info("No report yet. Ask the agent to compare the four parcel datasets.")
    else:
        if result.get("is_error"):
            st.error(result["content"])
        else:
            st.markdown(result["content"])

        trace = result.get("trace") or []
        if trace:
            with st.expander("Analysis activity"):
                for index, step in enumerate(trace, start=1):
                    with st.container(border=True):
                        st.markdown(f"**Step {index}**")
                        st.caption(f"Tool: `{step.get('tool', 'unknown')}`")
                        if step.get("arguments"):
                            st.json(step["arguments"])
                        preview = step.get("result_preview")
                        if preview:
                            st.code(preview, language="text")
    st.markdown("</div>", unsafe_allow_html=True)
