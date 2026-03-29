# Decision Co-Pilot

Decision Co-Pilot is a Streamlit app that reads a health insurance policy PDF and gives a strict claim decision based on that policy.

The demo flow is simple:

1. Upload one insurance policy PDF.
2. Paste one natural-language claim query.
3. Get an `ACCEPT` or `REJECT` decision.
4. See the explanation, likely payout, and policy evidence used for the answer.

This README is written as a full handoff guide so a teammate can set up the project on another Windows PC without needing to search for missing steps.

## 1. What This Project Uses

- Python
- Streamlit
- Groq API for Llama inference
- FAISS for retrieval
- Sentence Transformers for embeddings
- A local `.venv` virtual environment

## 2. Project Files

```text
hackrx-project/
|-- app.py                  # Main Streamlit entry point
|-- decision_core.py        # Core decision logic, parsing, retrieval, rule checks
|-- showcase_ui.py          # UI rendering helpers
|-- test_app.py             # Tests
|-- requirements.txt        # Python dependencies
|-- readme.md               # This guide
|-- .env                    # Local secrets file (you create this)
|-- .venv/                  # Local virtual environment (already used in this project)
|-- temp/                   # Temporary files and demo policy PDFs
```

## 3. Before You Start

Make sure the teammate has:

- Windows
- Python installed
- Internet connection for first-time dependency/model download
- A valid `GROQ_API_KEY`

## 4. How To Open The Project

1. Copy or clone the full project folder onto the other PC.
2. Open the folder in VS Code.
3. Open a PowerShell terminal in the project root.

The project root should be the folder that contains `app.py`, `requirements.txt`, and `readme.md`.

## 5. Check Python Installation

In PowerShell, run:

```powershell
python --version
```

If that does not work, try:

```powershell
py --version
```

If neither works, Python is not installed correctly and must be installed first.

## 6. Using The Existing `.venv`

This project already uses a local virtual environment folder called `.venv`.

If `.venv` is already present in the project folder, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal usually starts showing `(.venv)` at the beginning of the line.

If PowerShell blocks activation, use this command instead to temporarily allow it for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 7. If `.venv` Does Not Exist

If the teammate does not receive the `.venv` folder, create it manually.

Run:

```powershell
python -m venv .venv
```

Then activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If `python` does not work but `py` does, use:

```powershell
py -m venv .venv
```

## 8. Install Dependencies

After the virtual environment is active, install the required packages:

```powershell
pip install -r requirements.txt
```

If `pip` does not work, use:

```powershell
python -m pip install -r requirements.txt
```

Or, if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 9. Create The `.env` File

This app requires a Groq API key.

Create a file named `.env` in the project root.

The `.env` file must contain:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

Important:

- Do not add quotes unless needed.
- Do not name the file `.env.txt`.
- The file must be in the project root, beside `app.py`.

## 10. Run The App

Once the virtual environment is active and dependencies are installed, run:

```powershell
streamlit run app.py
```

If that does not work, run:

```powershell
python -m streamlit run app.py
```

Or directly with the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit should open a local browser tab, usually at:

```text
http://localhost:8501
```

## 11. How To Use The App

1. Start the app.
2. In the sidebar, upload a policy PDF.
3. Paste a claim query into the main text box.
4. Click `Send`.
5. Review the decision, payout, reasoning, and cited policy evidence.

The app is designed for one uploaded policy and one claim query at a time.

## 12. Example Demo Queries

These are safe sample prompts you can paste directly into the app.

Approved case:

```text
45-year-old male in Pune seeking reimbursement for accidental fracture treatment after an accident. Policy active for 3 years. Claim amount Rs 180000.
```

Rejected case:

```text
29-year-old female maternity delivery in Delhi. Policy purchased 5 months ago. Expected bill Rs 95000.
```

Sparse query:

```text
Hospitalization claim for cardiac treatment. Need to know whether the policy covers it and what payout is likely.
```

## 13. Recommended Way To Launch Every Time

Each time the teammate wants to run the project, these are the usual commands:

```powershell
cd path\to\hackrx-project
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

If activation causes trouble, use this direct form instead:

```powershell
cd path\to\hackrx-project
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 14. How To Run Tests

To verify the project is working correctly, run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

This runs the test suite in `test_app.py`.

## 15. Quick Health Check Commands

Use these if the teammate wants to check that the project is set up correctly.

Check Python in `.venv`:

```powershell
.\.venv\Scripts\python.exe --version
```

Check Streamlit is available:

```powershell
.\.venv\Scripts\python.exe -m streamlit --version
```

Check app files exist:

```powershell
dir
```

## 16. Common Problems And Fixes

### Problem: `GROQ_API_KEY is not set`

Reason:
The `.env` file is missing, in the wrong place, or the key name is incorrect.

Fix:

1. Make sure the file is named exactly `.env`
2. Make sure it is in the project root
3. Make sure it contains:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

Then stop and rerun the app.

### Problem: `streamlit` is not recognized

Fix:

Run with the virtual environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Problem: virtual environment activation is blocked

Fix:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Problem: packages are missing

Fix:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Problem: first run is slow

Reason:
Some dependencies and embedding-related assets may take time to initialize on first use.

Fix:
Wait for the first run to complete. Later runs are usually faster.

### Problem: uploaded PDF gives no useful answer

Reason:
The policy might have poor text extraction quality or unusual formatting.

Fix:

- Try a clearer policy PDF
- Retry with a more specific claim query
- Test first with the known SBI sample policy in `temp/`

## 17. Demo Policy File

The repository may include a sample file in the `temp` folder:

```text
temp/sbi_health_insurance_toc.pdf
```

This file is useful for:

- checking whether setup is working
- running the demo quickly
- verifying expected behavior with known prompt examples

## 18. What The App Actually Does Internally

At a high level:

1. Reads the uploaded PDF
2. Extracts policy text
3. Splits the text into searchable chunks
4. Builds a FAISS vector store
5. Parses the claim query into useful fields
6. Retrieves the most relevant policy clauses
7. Uses the Llama model through Groq for structured reasoning
8. Applies project-specific decision logic and guardrails
9. Returns a final `ACCEPT` or `REJECT` answer with evidence

## 19. Main Files To Edit

If the teammate needs to make changes later:

- `app.py` for app startup and page-level flow
- `showcase_ui.py` for UI layout and rendering
- `decision_core.py` for claim logic, parsing, retrieval, and decisions
- `test_app.py` for tests

## 20. Best Practice For Teammates

Ask the teammate to always do these in order:

1. Open project folder in terminal
2. Activate `.venv`
3. Confirm `.env` exists
4. Run `streamlit run app.py`

That avoids most setup issues.

## 21. One-Line Setup Summary

If the teammate only wants the shortest possible startup path:

```powershell
cd path\to\hackrx-project
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If activation fails, use:

```powershell
cd path\to\hackrx-project
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
