# Loan Application Risk Assistant

The first workflow stage reviews a synthetic applicant against the personal-loan
policy PDF. LangGraph runs input validation, policy retrieval, assessment,
output validation, and human review in sequence. Invalid input or an unsafe
assistant summary is rejected and recorded before human review; valid results
go to a human reviewer for the final decision.

## Setup

From the project directory, install dependencies and ensure Ollama is running
with the configured models:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull nomic-embed-text:latest
ollama pull llama3.2:3b
ollama pull llama3.2:latest
```

## Input guardrail agent

`Input_guardrail.py` provides the `llama3.2:latest` guidance agent and the
structured applicant-data validation used by `Loan_Risk_Assistant_Agent.py`.
Malformed or missing application data is rejected before RAG; missing policy
documents remain policy-check blockers and continue to output validation and
human review when the application data itself is valid.

```powershell
.\.venv\Scripts\python.exe .\Input_guardrail.py
```

## Output guardrail agent

`Output_guardrail.py` uses `llama3.2:latest` to review the generated applicant
summary. PII middleware redacts email, credit-card, phone, Aadhaar, and PAN
values in both the draft and the response. A summary that remains empty,
contains detected PII, or includes lending-decision language is rejected before
HITL.

```powershell
.\.venv\Scripts\python.exe .\Output_guardrail.py
```

The source policy is stored at `knowledge_base/Personal_Loan_Policy.pdf`.
The RAG chatbot script is at the project root and can be run with:

```powershell
.\.venv\Scripts\python.exe .\RAG_Loan_app.py
```

`Loan_Risk_Assistant_Agent.py` uses the guardrail helpers and agents from
`Input_guardrail.py` and `Output_guardrail.py`, and imports the policy loader
and retrieval helper from `RAG_Loan_app.py`. Valid applicant records proceed
through policy retrieval, assessment, output validation, and then HITL. This
does not launch either guardrail script's interactive loop or the RAG chatbot.

The demo inputs are generated under `inputs/`:

```powershell
.\.venv\Scripts\python.exe .\examples\create_demo_inputs.py
```

The fixtures cover five scenarios: APP001 and APP004 pass the policy checks,
APP002 is missing Bank Statements and Income Declaration, APP003 is salaried
but has monthly income below the policy minimum, and APP005 has no application
document. APP004 has distinct synthetic applicant details and replaces the
obsolete AP001 source document. The workbook includes a `Risk Assessment`
column with `No Risk`, `Low Risk`, or `High Risk`; regenerating fixtures adds
missing columns and applicant rows without resetting existing review statuses.

By default, the assistant reads every applicant from
`inputs/Input_Loan_Applicants.xlsx` and reviews them one at a time. It writes
results to `output/Updated_Loan_Applicants.xlsx`, leaving the input workbook
unchanged. The output
includes `Loan_risk_assistant_comment`, `Reviewer Decision`, `Risk Assessment`,
and `Status` columns for each applicant. The assistant comment captures the
summary, recommendation, and risk indicators. Existing output decisions are
preserved for applicants not reviewed during the current run.

```powershell
.\.venv\Scripts\python.exe .\Loan_Risk_Assistant_Agent.py
```

To review only one applicant:

```powershell
.\.venv\Scripts\python.exe .\Loan_Risk_Assistant_Agent.py --applicant-id APP004
```

The code's demo applicant ID list is `APP001` through `APP005`. Display it with:

```powershell
.\.venv\Scripts\python.exe .\Loan_Risk_Assistant_Agent.py --list-demo-applicants
```

After both guardrail stages pass, the reviewer chooses `APPROVE`, `REJECT`, or
`REQUEST_INFO` for each applicant.
Failed or pending policy checks recommend rejection or more information. If
the reviewer approves despite a failed or pending policy check, the override
is warned about and recorded as `High Risk`. An approval with all policy
checks passed is `No Risk`; rejected or information-requested applications
are `Low Risk`. APP005's missing application document fails input validation
and is automatically recorded as rejected without entering HITL; this does
not stop reviews of other applicants. Decisions, assistant comments, and risk
assessments are also appended to `inputs/review_history.jsonl`.

Use `--applicant-id`, `--workbook`, `--output-workbook`, `--application`, and
`--history` to select other records or paths. The source Excel sheet needs
`Applicant_ID`, `Full_Name`, and `Age` columns. The output workbook is created
with `Status`, `Loan_risk_assistant_comment`, `Reviewer Decision`, and
`Risk Assessment` columns if needed. The application DOCX needs the labeled
applicant fields and document-status lines shown in the generated APP001
example.

The index is persisted in `Chroma/` and rebuilt when the policy PDF changes.
The applicant and document fixtures are synthetic and intended only for
demonstration; replace them with authorized, validated application records
before operational use.
