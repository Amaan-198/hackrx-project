# Decision Co-Pilot

**AI-Powered Insurance Claim Assistant**

---

## 🚀 Project Overview

Decision Co-Pilot is a comprehensive, end-to-end solution for automating and explaining health insurance claim decisions. It combines Retrieval-Augmented Generation (RAG) with a rule-based engine to:

* **Understand** any policy document (PDF, Word, or email) by extracting clauses, limits, waiting periods, and exclusions.
* **Interpret** natural language queries (e.g., “Cataract surgery, policy 1 year old”) with precise parsing of age, condition, policy duration, and sum insured.
* **Decide** claims automatically (Approved/Rejected/Needs Clarification) with a multi-factor confidence score.
* **Explain** every decision with structured JSON output, quoting exact policy clauses (page and section) and detailed reasoning steps.
* **Scale** via batch processing and offer real-time insights through an analytics dashboard.

This project was built for the HackRx 6.0 Hackathon by Bajaj Finserv Health, competing against over 30,000 teams. What sets us apart is our commitment to **complete transparency**—no black-box outputs: judges and auditors see both the ‘what’ and the ‘why’.

## 📂 Repository Structure

```
decision-co-pilot/
├── app.py                  # Streamlit application entry point
├── query_parser.py         # Extracts structured data (age, condition, duration, etc.) from queries
├── rule_engine.py          # Validates claims against policy rules (waiting periods, exclusions)
├── confidence_calc.py      # Calculates a comprehensive confidence score
├── utils.py                # Shared helper functions
├── requirements.txt        # Python package dependencies
├── README.md               # This detailed project documentation
├── .gitignore              # Files and folders to ignore for Git
└── data/
    └── sample_policy.pdf   # Optional: sample SBI policy for demos
```

## 🔧 Key Features

1. **Policy Ingestion & Rule Extraction**

   * Upload any policy PDF.
   * `InsuranceRuleEngine` uses regex and spaCy to parse waiting periods, age limits, exclusions, co-pay rules, sub-limits, and more.
   * Extracted rules override default settings automatically.

2. **Smart Query Parsing**

   * `QueryParser` class tokenizes and analyzes user queries to identify:

     * Age, gender, condition (e.g., diabetes, cataract)
     * Policy duration and unit (months/years)
     * Sum insured or amount mentioned
     * Completeness score (to flag missing details)

3. **RAG-Based Retrieval & LLM Reasoning**

   * Documents are split into chunks (1,000 characters, with overlap) and embedded using `all-MiniLM-L6-v2`.
   * FAISS vector store enables semantic search for the top-K relevant passages.
   * A quantized Llama-3 model (via Ollama) generates structured answers based on retrieved context.

4. **Rule-Based Validation**

   * Validates parsed claims against policy-specific rules:

     * Waiting periods (e.g., 30 days for all illnesses, 2 years for cataract)
     * Exclusions (e.g., cosmetic surgery, fertility treatments)
     * Age limits, co-pay requirements
   * Overrides any “Approved” outputs if violations are detected and logs violations explicitly.

5. **Multi-Factor Confidence Scoring**

   * Combines factors:

     * Query completeness (0–1)
     * Document relevance (overlap measure)
     * Rule validation impact (penalties for violations)
     * LLM self-reported confidence
     * Consistency checks (e.g., amount-decision mismatch)
   * Provides a weighted final confidence score plus detailed factor breakdown.

6. **Explainable JSON Output**

   * Returns JSON with:

     * `Decision`: APPROVED / REJECTED / REQUIRES\_CLARIFICATION
     * `Amount`: numeric payout or “Not specified” strings
     * `Justification`: exact clause text and page references
     * `Confidence`: final confidence (0.0–1.0)
     * `ReasoningSteps`: list of step-by-step analyses
     * `SourcePages`: page numbers of supporting evidence
     * `RuleViolations`: any policy rule breaches

7. **Batch Processing**

   * Upload a text or CSV file of multiple queries.
   * Processes all queries concurrently, showing progress and exporting full JSON results.

8. **Analytics Dashboard**

   * Displays key metrics: total queries, approval rate, avg. confidence, avg. payout.
   * Plots: decision distribution, confidence histogram, completeness levels, violation impact trends, and time-series of query volume vs. confidence.

9. **Polished UI with Streamlit**

   * Tabbed interface: Single Query, Batch Processing, Analytics.
   * Custom CSS for headers, confidence badges, and reasoning boxes.
   * Robust error handling and fallback JSON parsing ensure stability.

## 🏁 Quickstart Guide

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/decision-co-pilot.git
   cd decision-co-pilot
   ```
2. **Install dependencies**

   ```bash
   python3 -m venv env
   source env/bin/activate    # Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Run the app**

   ```bash
   streamlit run app.py
   ```
4. **Demo sample queries**

   * Upload `data/sample_policy.pdf`
   * Try:

     * “I was hospitalized for diabetes, policy 8 months old, sum insured ₹300,000” → *Rejected* citing Clause 2 (page 6)
     * “Cataract surgery, policy 1 year old, sum insured ₹300,000” → *Approved* citing Clause 4 (page 2)
   * Switch to Batch tab, upload a CSV of claims, and watch them process.
   * Explore Analytics for insights.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

*Last updated: August 2025 — built for HackRx 6.0 by Bajaj Finserv Health.*
