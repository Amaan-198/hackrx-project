from html import escape
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from tech_lab import build_tech_demo_manifest


TECH_LAB_BASE_URL = "http://localhost:8502"


def initialize_session_state(defaults: Dict[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    st.session_state.latest_query = ""
    st.session_state.latest_result = None


def render_user_bar() -> None:
    col_left, col_right = st.columns([4, 1])
    with col_right:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:flex-end;gap:0.6rem;margin:0.4rem 0;">'
            '<span style="font-size:0.78rem;color:#7b898f;">demo@decision.local</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Sign out", key="logout_btn", use_container_width=False):
            st.markdown(
                '<meta http-equiv="refresh" content="0; url=http://localhost:8502">',
                unsafe_allow_html=True,
            )
            st.stop()
    with col_left:
        st.caption("")


def render_hero(has_document: bool) -> None:
    subtitle = (
        "Upload one policy, paste one claim query, and get a strict accept-or-reject answer."
        if has_document
        else "A cleaner demo flow with one prompt box, copyable example queries, and a chat-style answer."
    )
    st.markdown(
        f"""
<div class="hero-shell">
    <div class="hero-eyebrow">Decision Co-Pilot</div>
    <div class="hero-title">Insurance decisions, presented like a conversation.</div>
    <p class="hero-subtitle">{escape(subtitle)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_view() -> None:
    st.markdown(
        """
<div class="landing-shell">
    <div class="section-label">Start here</div>
    <h3>Upload a policy PDF from the sidebar</h3>
    <p>The screen stays focused after that: one prompt composer, one prompt library, one answer thread.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar() -> Optional[Any]:
    with st.sidebar:
        st.subheader("Policy PDF")
        uploaded_file = st.file_uploader("Upload document", type="pdf")
        if st.button("Reset analysis", use_container_width=True):
            reset_analysis_state()

        if st.session_state.get("document_profile"):
            st.markdown("---")
            st.caption("Loaded policy")
            st.write(st.session_state.document_profile.get("document_name", "Unnamed document"))
            st.caption(f"{st.session_state.document_profile.get('page_count', 0)} pages")

        st.markdown("---")
        with st.expander("Technology lab", expanded=False):
            st.caption("Run `python backend_api.py` and open the local demos on port 8502.")
            for demo in build_tech_demo_manifest():
                st.markdown(f"[{demo['title']}]({TECH_LAB_BASE_URL}{demo['url']})")

        return uploaded_file


def render_policy_snapshot(policy_radar: Dict[str, Any], extracted_rules: Dict[str, Any]) -> None:
    waits = extracted_rules.get("waiting_periods", {})
    chips = []
    if waits.get("general"):
        chips.append(f"General wait {waits['general']}m")
    if waits.get("pre_existing"):
        chips.append(f"Pre-existing {waits['pre_existing']}m")
    if waits.get("maternity"):
        chips.append(f"Maternity {waits['maternity']}m")
    if extracted_rules.get("exclusions"):
        chips.append(f"{len(extracted_rules['exclusions'])} exclusions")
    summary = policy_radar.get("missing_signals", [])[:1] or ["Policy signals extracted and ready."]
    st.markdown(
        f"""
<div class="policy-strip">
    <div class="chip-row">{''.join(f"<span class='meta-chip'>{escape(chip)}</span>" for chip in chips)}</div>
    <p>{escape(summary[0])}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_query_studio(example_queries: List[Dict[str, str]]) -> Tuple[bool, str]:
    del example_queries
    st.markdown(
        """
<div class="composer-shell">
    <div class="section-label">Prompt composer</div>
    <h3>Write the claim exactly the way you would say it in a demo.</h3>
    <p>The engine will pull out age, treatment, policy tenure, location, and amount from the sentence itself.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("single_query_form", clear_on_submit=False):
        query = st.text_area(
            "Claim query",
            key="claim_query",
            height=140,
            placeholder="Paste a claim query here...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", use_container_width=True)
    return submitted, query


def render_prompt_library(example_queries: List[Dict[str, str]]) -> None:
    st.markdown(
        """
<div class="library-heading">
    <div class="section-label">Prompt library</div>
    <h3>Try another prompt</h3>
    <p>Copy any prompt below and paste it into the query box.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    for example in example_queries:
        st.markdown(f"<div class='prompt-title'>{escape(example['title'])}</div>", unsafe_allow_html=True)
        st.code(example["query"], language=None)


def render_session_break() -> None:
    st.markdown(
        """
<div class="session-break">
    <div class="session-rule"></div>
    <div class="session-note">Continue the demo</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_response_placeholder() -> None:
    st.markdown(
        """
<div class="response-shell empty">
    <div class="response-header">
        <span class="response-kicker">Response</span>
    </div>
    <p>Send a claim query to see the policy-backed decision appear here.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_response_result(result: Dict[str, Any], format_currency) -> None:
    decision = result.get("decision", "REJECTED")
    decision_label = "ACCEPT" if decision == "APPROVED" else "REJECT"
    amount = format_currency(result.get("amount", 0))
    pages = ", ".join(str(page) for page in result.get("source_pages", [])) or "No pages cited"
    reasoning_items = result.get("reasoning_steps", [])[:3]
    evidence_items = [f"Page {item['page']}: {item['snippet']}" for item in result.get("source_evidence", [])[:3]]
    rule_items = result.get("rule_violations", []) or result.get("rule_validation", {}).get("applicable_rules", [])

    def _html_items(items: List[str]) -> str:
        if not items:
            return "<p class='inline-note'>No extra evidence extracted.</p>"
        return "".join(f"<li>{escape(item)}</li>" for item in items)

    st.markdown(
        f"""
<div class="response-shell">
    <div class="response-header">
        <span class="response-kicker">Response</span>
        <div class="answer-topline">
            <span class="decision-pill {'accept' if decision == 'APPROVED' else 'reject'}">{escape(decision_label)}</span>
            <span class="payout-pill">{escape(amount)}</span>
        </div>
    </div>
    <p class="answer-copy">{escape(result.get('justification', 'No justification provided'))}</p>
    <div class="answer-meta">Pages {escape(pages)}</div>
    <div class="answer-grid">
        <div class="answer-section">
            <div class="answer-section-title">Why</div>
            <ul class="thread-list">{_html_items(reasoning_items)}</ul>
        </div>
        <div class="answer-section">
            <div class="answer-section-title">Policy evidence</div>
            <ul class="thread-list">{_html_items(evidence_items or rule_items[:4])}</ul>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
