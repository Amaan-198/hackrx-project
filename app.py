import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

import showcase_ui as ui
from decision_core import (
    BatchProcessor,
    ClaimDecisionEngine,
    ConfidenceCalculator,
    InsuranceRuleEngine,
    MODEL_NAME,
    QueryParser,
    build_action_plan,
    build_claim_query,
    build_policy_radar,
    clean_json_response,
    create_enhanced_qa_chain,
    create_enhanced_vector_store,
    create_query_parser_llm,
    extract_source_evidence,
    format_currency,
    initialize_embedding_model,
    process_enhanced_response,
)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

TEMP_DIR = "temp"
EXAMPLE_QUERIES = [
    {
        "title": "Approved",
        "query": (
            "45-year-old male in Pune seeking reimbursement for accidental fracture treatment after an accident. "
            "Policy active for 3 years. Claim amount Rs 180000."
        ),
    },
    {
        "title": "Rejected",
        "query": (
            "29-year-old female maternity delivery in Delhi. "
            "Policy purchased 5 months ago. Expected bill Rs 95000."
        ),
    },
    {
        "title": "Sparse query",
        "query": (
            "Hospitalization claim for cardiac treatment. Need to know whether the policy covers it "
            "and what payout is likely."
        ),
    },
]

st.set_page_config(page_title="Decision Co-Pilot", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:wght@400;600&display=swap');
:root { --panel: rgba(255,252,247,0.96); --ink:#15262c; --muted:#5f6d73; --accent:#0e6c67; --accent-soft:rgba(14,108,103,0.10); --line:rgba(20,37,44,0.10); --shadow:0 16px 48px rgba(20,37,44,0.07); }
html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #fbf7f1 0%, #f2ece3 100%); color:var(--ink); font-family:"Space Grotesk","Segoe UI",sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #14252c 0%, #1c3942 100%); border-right:1px solid rgba(255,255,255,0.06); }
[data-testid="stSidebar"] * { color:#f8f4ed; }
.main .block-container { max-width:960px; padding-top:2rem; padding-bottom:3rem; }
div[data-testid="stForm"], .composer-shell, .landing-shell { background:var(--panel); border:1px solid var(--line); border-radius:24px; box-shadow:var(--shadow); }
div[data-testid="stForm"] { padding:1rem 1rem 0.35rem 1rem; }
.hero-shell { position:relative; overflow:hidden; padding:1.9rem 2rem; border-radius:30px; background:linear-gradient(135deg, rgba(20,37,44,0.97) 0%, rgba(14,108,103,0.90) 100%); box-shadow:0 24px 70px rgba(20,37,44,0.14); color:#fff9f2; margin-bottom:1rem; }
.hero-shell::after { content:""; position:absolute; inset:auto -4rem -5rem auto; width:18rem; height:18rem; background:radial-gradient(circle, rgba(227,122,69,0.34), transparent 68%); }
.hero-eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.76rem; color:rgba(255,249,242,.78); margin-bottom:.8rem; }
.hero-title { font-family:"Source Serif 4", Georgia, serif; font-size:clamp(2.4rem, 4vw, 3.5rem); line-height:.98; margin:0 0 .7rem 0; }
.hero-subtitle { max-width:38rem; font-size:1rem; color:rgba(255,249,242,.84); margin:0; }
.composer-shell, .landing-shell { padding:1.2rem 1.25rem; }
.composer-shell h3, .landing-shell h3 { margin:0 0 .35rem 0; font-size:1.02rem; }
.composer-shell p, .landing-shell p, .library-heading p, .policy-strip p { margin:0; color:var(--muted); line-height:1.55; }
.section-label { display:inline-flex; align-items:center; padding:.35rem .75rem; border-radius:999px; font-size:.74rem; text-transform:uppercase; letter-spacing:.12em; background:var(--accent-soft); color:var(--accent); margin-bottom:.75rem; }
.policy-strip { display:flex; flex-wrap:wrap; justify-content:space-between; gap:.9rem; align-items:center; padding:.8rem .95rem; border:1px solid var(--line); border-radius:18px; background:rgba(255,252,247,0.68); margin:2rem 0 0 0; }
.meta-chip { padding:.45rem .7rem; border-radius:999px; background:rgba(20,37,44,.06); color:var(--ink); font-size:.82rem; display:inline-flex; }
.chip-row { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.85rem; }
.stButton > button, .stFormSubmitButton > button { border-radius:999px; border:1px solid #0e6c67; background:#0e6c67; color:#fff9f2; font-weight:600; padding:.72rem 1rem; transition:transform .18s ease, box-shadow .18s ease, background .18s ease; box-shadow:0 10px 28px rgba(14,108,103,.14); }
.stButton > button:hover, .stFormSubmitButton > button:hover { transform:translateY(-1px); box-shadow:0 14px 30px rgba(14,108,103,.16); background:#0c5b57; }
[data-testid="stSidebar"] .stButton > button { background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.08); color:#f8f4ed; box-shadow:none; }
[data-testid="stSidebar"] .stButton > button:hover { background:rgba(255,255,255,0.12); box-shadow:none; }
.stTextArea textarea, .stTextInput input { border-radius:18px !important; color:var(--ink) !important; background:#fffdf9 !important; }
.stTextArea textarea::placeholder { color:#7a878d !important; }
[data-testid="stCodeBlock"] { border-radius:18px; border:1px solid var(--line); overflow:hidden; margin:.3rem 0 1.2rem 0; }
[data-testid="stCodeBlock"] pre { background:#f6f1e8 !important; color:#1b2a30 !important; }
[data-testid="stCodeBlock"] code, [data-testid="stCodeBlock"] span { color:#1b2a30 !important; }
[data-testid="stCodeBlock"] button { color:#56656b !important; }
.library-heading { margin-top:2.4rem; }
.library-heading h3 { margin:0 0 .25rem 0; font-size:1.05rem; color:#203238; }
.prompt-title { font-size:.86rem; font-weight:700; color:#2b3c42; margin:1rem 0 .35rem 0; }
.response-shell { background:#fffdf8; border:1px solid var(--line); box-shadow:var(--shadow); border-radius:26px; padding:1.2rem 1.25rem; margin-top:1rem; }
.response-shell.empty { color:var(--muted); }
.response-header { display:flex; flex-wrap:wrap; justify-content:space-between; gap:.8rem; align-items:center; margin-bottom:.9rem; }
.response-kicker { font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:#6b7a80; }
.answer-topline { display:flex; flex-wrap:wrap; gap:.7rem; align-items:center; }
.decision-pill, .payout-pill { display:inline-flex; align-items:center; border-radius:999px; padding:.42rem .8rem; font-size:.82rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.decision-pill.accept { background:rgba(14,108,103,.10); color:#0e6c67; }
.decision-pill.reject { background:rgba(188,74,54,.10); color:#a13d2a; }
.payout-pill { background:rgba(20,37,44,.06); color:#15262c; }
.answer-copy { font-size:1rem; color:#15262c; margin-bottom:.8rem !important; }
.answer-meta { color:#617077; font-size:.88rem; margin-bottom:.9rem; }
.meta-divider { opacity:.45; margin:0 .35rem; }
.answer-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:1rem; }
.answer-section { padding:.95rem 1rem; border-radius:18px; background:#f8f3eb; border:1px solid rgba(20,37,44,.06); }
.answer-section-title { font-size:.8rem; letter-spacing:.12em; text-transform:uppercase; color:#0e6c67; margin-bottom:.45rem; }
.thread-list { margin:0; padding-left:1rem; color:#304046; }
.thread-list li { margin-bottom:.38rem; }
.inline-note { color:#617077; font-size:.92rem; }
.session-break { display:flex; align-items:center; gap:1rem; margin:2.1rem 0 1.3rem 0; }
.session-rule { flex:1; height:1px; background:linear-gradient(90deg, rgba(20,37,44,0.16) 0%, rgba(20,37,44,0.06) 52%, rgba(20,37,44,0) 100%); }
.session-note { font-size:.76rem; letter-spacing:.14em; text-transform:uppercase; color:#7b898f; white-space:nowrap; }
@media (max-width: 900px) { .answer-grid { grid-template-columns:1fr; } }
::selection { background:#d9e6ff; color:#15262c; }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it to your .env file and restart.")
        st.stop()

    ui.initialize_session_state(
        {
            "query_parser": QueryParser(create_query_parser_llm()),
            "rule_engine": InsuranceRuleEngine(),
            "confidence_calc": ConfidenceCalculator(),
            "latest_query": "",
            "latest_result": None,
            "document_profile": {},
            "policy_radar": {},
            "_file_hash": None,
            "document_name": "",
            "claim_query": "",
        }
    )

    uploaded_file = ui.render_sidebar()
    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if st.session_state.get("_file_hash") != file_hash:
            os.makedirs(TEMP_DIR, exist_ok=True)
            file_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(file_path, "wb") as handle:
                handle.write(uploaded_file.getvalue())
            vector_store, page_mapping, extracted_rules, document_profile = create_enhanced_vector_store(file_path, file_hash)
            qa_chain, output_parser = create_enhanced_qa_chain(vector_store, page_mapping)
            st.session_state.document_name = uploaded_file.name
            st.session_state.document_profile = document_profile
            st.session_state.rule_engine.extracted_rules = extracted_rules
            st.session_state.policy_radar = build_policy_radar(document_profile["policy_text"], extracted_rules)
            st.session_state.decision_engine = ClaimDecisionEngine(
                qa_chain,
                output_parser,
                st.session_state.query_parser,
                st.session_state.rule_engine,
                st.session_state.confidence_calc,
                page_mapping,
            )
            st.session_state._file_hash = file_hash
            ui.reset_analysis_state()

    has_document = "decision_engine" in st.session_state
    ui.render_hero(has_document)
    if not has_document:
        ui.render_landing_view()
        return

    submitted, query = ui.render_query_studio(EXAMPLE_QUERIES)

    if submitted:
        query = query.strip()
        if not query:
            st.warning("Add a claim query first.")
        else:
            with st.spinner("Analyzing claim..."):
                st.session_state.latest_query = query
                st.session_state.latest_result = st.session_state.decision_engine.process_single_query(query)
                st.rerun()

    if st.session_state.get("latest_result"):
        ui.render_response_result(st.session_state.latest_result, format_currency)
        ui.render_session_break()
    else:
        ui.render_response_placeholder()

    ui.render_prompt_library(EXAMPLE_QUERIES)
    ui.render_policy_snapshot(st.session_state.policy_radar, st.session_state.rule_engine.extracted_rules)


if __name__ == "__main__":
    main()
