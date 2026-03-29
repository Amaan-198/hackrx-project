# Decision Co-Pilot

Decision Co-Pilot is a Streamlit showcase app for explaining health insurance claim decisions against a real policy PDF. The experience is built for demos: upload one policy, run one claim, inspect the policy radar, and use a what-if lab to show how the outcome changes when claim facts change.

## What the app does

- Extracts waiting periods, age limits, and exclusions from the uploaded policy.
- Parses claim narratives into structured facts with a hybrid parser that uses regex first and Llama only when the claim is incomplete or ambiguous.
- Retrieves the most relevant policy passages with FAISS and asks Groq Llama for a structured claim decision.
- Applies deterministic guardrails so obvious rule violations override overly optimistic model outputs.
- Surfaces visible demo features on screen: decision spotlight, policy radar, scenario lab, and action plan / appeal draft.

## Project structure

```text
hackrx-project/
|-- app.py             # Streamlit entry point and showcase flow
|-- decision_core.py   # Parsing, rule extraction, retrieval, decision engine
|-- showcase_ui.py     # UI rendering helpers
|-- test_app.py        # Unit and policy-regression tests
|-- temp/              # Local demo files such as the SBI sample policy PDF
```

## Demo flow

1. Upload the policy PDF.
2. Pick a demo scenario or type a claim story.
3. Run the live decision.
4. Show the cited pages, rule checks, and source evidence.
5. Open the scenario lab to prove the decision moves when the facts move.

The included SBI demo policy currently gives a clean presentation path:

- Accidental fracture after 3 years of policy tenure: approved
- Maternity delivery after 5 months: rejected
- Sparse cardiac query with missing details: requires clarification

## Setup

1. Create and activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add `GROQ_API_KEY` to `.env`.
4. Run the app:

```bash
streamlit run app.py
```

## Testing

Run the full suite with your local `.venv`:

```bash
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

The test suite includes regression coverage for the real SBI demo PDF in `temp/sbi_health_insurance_toc.pdf` so policy extraction bugs are caught early.
