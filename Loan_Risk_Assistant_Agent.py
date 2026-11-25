import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from docx import Document
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from openpyxl import Workbook, load_workbook
from pydantic import BaseModel, ConfigDict, Field
from Input_guardrail import validate_applicant_input
from Output_guardrail import (
    create_output_guardrail_agent,
    validate_output_with_agent,
)
from RAG_Loan_app import find_policy_pdf, load_vector_store, retrieve_policy_context


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "output"
SOURCE_WORKBOOK = INPUT_DIR / "Input_Loan_Applicants.xlsx"
OUTPUT_WORKBOOK = OUTPUT_DIR / "Updated_Loan_Applicants.xlsx"
EMBEDDING_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "llama3.2:3b"
DEMO_APPLICANTS = ("APP001", "APP002", "APP003", "APP004", "APP005")

REQUIRED_DOCUMENTS = (
    "Aadhaar Card",
    "PAN Card",
    "Salary Slips",
    "Bank Statements",
    "Employment Proof",
    "Address Proof",
    "Photograph",
    "Income Declaration",
)
TENURE_RATES = {12: 12.0, 24: 12.5, 36: 13.0, 48: 13.5, 60: 14.0}
PROFILE_FIELDS = (
    "age",
    "employment_status",
    "company_registered",
    "monthly_income",
    "requested_amount",
    "tenure_months",
    "annual_interest_rate",
    "declared_emi",
    "applicant_confirmation",
)


class DocumentStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submitted: bool
    verified: bool


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicant_id: str = Field(min_length=1)
    full_name: str = Field(min_length=1)
    age: int = Field(gt=0)
    employment_status: str = Field(min_length=1)
    company_registered: bool
    monthly_income: float = Field(gt=0)
    requested_amount: float = Field(gt=0)
    tenure_months: int = Field(gt=0)
    annual_interest_rate: float = Field(gt=0)
    declared_emi: float = Field(gt=0)
    applicant_confirmation: bool
    documents: dict[str, DocumentStatus]


class AssessmentNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)


class WorkflowState(TypedDict, total=False):
    applicant_id: str
    input_workbook_path: str
    output_workbook_path: str
    application_path: str
    history_path: str
    input_errors: list[str]
    input_valid: bool
    applicant: dict | None
    applicant_identity: dict
    application_missing: bool
    policy_context: str
    source_pages: list[int]
    narrative: dict
    checks: list[str]
    blockers: list[str]
    missing_documents: list[str]
    risk_indicators: list[str]
    recommendation: str
    emi_calculated: float | None
    data_completeness_confidence: float
    approval_allowed: bool
    reviewer_id: str
    decision: str
    final_status: str
    risk_assessment: str
    assistant_comment: str
    output_valid: bool
    output_errors: list[str]


def parse_bool(value: str, field_name: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise ValueError(f"{field_name} must be Yes or No, got {value!r}")


def parse_number(value: str, field_name: str) -> float:
    normalized = re.sub(r"[^0-9.\-]", "", value)
    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain a number, got {value!r}") from exc


def required_cell(row: tuple, column: int, field_name: str) -> str:
    value = row[column]
    if value is None or not str(value).strip():
        raise ValueError(f"Applicant workbook is missing {field_name}")
    return str(value).strip()


def read_applicant(workbook_path: Path, application_path: Path, applicant_id: str) -> Applicant:
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Applicant workbook not found: {workbook_path}")
    if not application_path.is_file():
        raise FileNotFoundError(f"Application document not found: {application_path}")

    workbook = load_workbook(workbook_path)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    headers = next(rows, None)
    if not headers:
        raise ValueError(f"Applicant workbook is empty: {workbook_path}")
    header_map = {
        str(value).strip().casefold(): index
        for index, value in enumerate(headers)
        if value is not None
    }
    if not {"applicant_id", "full_name", "age", "status"}.issubset(header_map):
        raise ValueError(
            "Applicant workbook must contain Applicant_ID, Full_Name, Age, and Status columns"
        )

    matches = [
        row
        for row in rows
        if str(row[header_map["applicant_id"]]).strip().casefold()
        == applicant_id.casefold()
    ]
    if not matches:
        raise ValueError(f"Applicant {applicant_id!r} was not found in {workbook_path}")
    if len(matches) > 1:
        raise ValueError(f"Applicant ID {applicant_id!r} occurs more than once in the workbook")
    applicant_row = matches[0]
    applicant_name = required_cell(
        applicant_row,
        header_map["full_name"],
        "Full_Name",
    )
    applicant_age = applicant_row[header_map["age"]]
    if applicant_age is None:
        raise ValueError("Applicant workbook is missing Age")

    paragraphs = [paragraph.text.strip() for paragraph in Document(application_path).paragraphs]
    fields: dict[str, str] = {}
    documents: dict[str, dict[str, bool]] = {}
    for paragraph in paragraphs:
        if paragraph.startswith("DOCUMENT:"):
            match = re.fullmatch(
                r"DOCUMENT:\s*(.*?)\s*\|\s*Submitted:\s*(Yes|No)"
                r"\s*\|\s*Verified:\s*(Yes|No)",
                paragraph,
                flags=re.IGNORECASE,
            )
            if not match:
                raise ValueError(f"Invalid document-status line: {paragraph!r}")
            name, submitted, verified = match.groups()
            documents[name] = {
                "submitted": parse_bool(submitted, f"{name} submitted"),
                "verified": parse_bool(verified, f"{name} verified"),
            }
        elif ":" in paragraph:
            key, value = paragraph.split(":", 1)
            fields[key.strip().casefold()] = value.strip()

    doc_applicant_id = fields.get("applicant id")
    if doc_applicant_id is None:
        raise ValueError(f"Application document is missing Applicant ID: {application_path}")
    if doc_applicant_id.casefold() != applicant_id.casefold():
        raise ValueError(
            f"Application document belongs to {doc_applicant_id!r}, not {applicant_id!r}"
        )

    def required_field(name: str) -> str:
        try:
            return fields[name.casefold()]
        except KeyError as exc:
            raise ValueError(f"Application document is missing required field: {name}") from exc

    raw_applicant = {
        "applicant_id": required_cell(
            applicant_row,
            header_map["applicant_id"],
            "Applicant_ID",
        ),
        "full_name": applicant_name,
        "age": applicant_age,
        "employment_status": required_field("Employment Status"),
        "company_registered": parse_bool(
            required_field("Company Registered"), "Company Registered"
        ),
        "monthly_income": parse_number(required_field("Monthly Income"), "Monthly Income"),
        "requested_amount": parse_number(
            required_field("Requested Loan Amount"), "Requested Loan Amount"
        ),
        "tenure_months": int(
            parse_number(required_field("Tenure Months"), "Tenure Months")
        ),
        "annual_interest_rate": parse_number(
            required_field("Annual Interest Rate"), "Annual Interest Rate"
        ),
        "declared_emi": parse_number(required_field("Declared EMI"), "Declared EMI"),
        "applicant_confirmation": parse_bool(
            required_field("Applicant Confirmation"), "Applicant Confirmation"
        ),
        "documents": documents,
    }
    return Applicant.model_validate(raw_applicant)


def read_applicant_identity(workbook_path: Path, applicant_id: str) -> dict[str, str | int]:
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Applicant workbook not found: {workbook_path}")

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    headers = next(rows, None)
    if not headers:
        raise ValueError(f"Applicant workbook is empty: {workbook_path}")
    header_map = {
        str(value).strip().casefold(): index
        for index, value in enumerate(headers)
        if value is not None
    }
    if not {"applicant_id", "full_name", "age"}.issubset(header_map):
        raise ValueError(
            "Applicant workbook must contain Applicant_ID, Full_Name, and Age columns"
        )
    matches = [
        row
        for row in rows
        if str(row[header_map["applicant_id"]]).strip().casefold()
        == applicant_id.casefold()
    ]
    workbook.close()
    if not matches:
        raise ValueError(f"Applicant {applicant_id!r} was not found in {workbook_path}")
    if len(matches) > 1:
        raise ValueError(f"Applicant ID {applicant_id!r} occurs more than once in the workbook")

    row = matches[0]
    age = row[header_map["age"]]
    if age is None:
        raise ValueError(f"Applicant {applicant_id!r} is missing Age in {workbook_path}")
    return {
        "applicant_id": required_cell(row, header_map["applicant_id"], "Applicant_ID"),
        "full_name": required_cell(row, header_map["full_name"], "Full_Name"),
        "age": age,
    }


def list_applicant_ids(workbook_path: Path) -> list[str]:
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Applicant workbook not found: {workbook_path}")

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    sheet = workbook.active
    header_row = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), None)
    if not header_row:
        raise ValueError(f"Applicant workbook is empty: {workbook_path}")
    headers = {
        str(value).strip().casefold(): index
        for index, value in enumerate(header_row)
        if value is not None
    }
    id_column = headers.get("applicant_id")
    if id_column is None:
        raise ValueError("Applicant workbook must contain an Applicant_ID column")

    applicant_ids: list[str] = []
    seen_ids: set[str] = set()
    for row in sheet.iter_rows(min_row=2, values_only=True):
        value = row[id_column] if id_column < len(row) else None
        if value is None or not str(value).strip():
            continue
        applicant_id = str(value).strip()
        normalized_id = applicant_id.casefold()
        if normalized_id in seen_ids:
            raise ValueError(f"Applicant ID {applicant_id!r} occurs more than once")
        seen_ids.add(normalized_id)
        applicant_ids.append(applicant_id)

    workbook.close()
    if not applicant_ids:
        raise ValueError(f"No applicants were found in {workbook_path}")
    return applicant_ids


def ensure_output_workbook(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Applicant input workbook not found: {input_path}")

    input_book = load_workbook(input_path, data_only=True)
    input_sheet = input_book.active
    input_rows = list(input_sheet.iter_rows(values_only=True))
    input_book.close()
    if not input_rows or not input_rows[0]:
        raise ValueError(f"Applicant input workbook is empty: {input_path}")

    source_headers = [
        str(value).strip() if value is not None else ""
        for value in input_rows[0]
    ]
    source_header_map = {
        header.casefold(): index
        for index, header in enumerate(source_headers)
        if header
    }
    if "applicant_id" not in source_header_map:
        raise ValueError("Applicant input workbook must contain an Applicant_ID column")

    protected_columns = {
        "status",
        "reviewer decision",
        "risk assessment",
        "loan_risk_assistant_comment",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_new_workbook = not output_path.exists()
    if is_new_workbook:
        output_book = Workbook()
        output_sheet = output_book.active
        output_sheet.title = "Applicants"
        output_sheet.append(source_headers)
        if "status" in source_header_map:
            output_sheet.cell(1, source_header_map["status"] + 1, "Status")
            for row_index in range(2, len(input_rows) + 1):
                row_values = list(input_rows[row_index - 1])
                for header, source_column in source_header_map.items():
                    if header in protected_columns:
                        row_values[source_column] = None
                output_sheet.append(row_values)
        else:
            for row_values in input_rows[1:]:
                output_sheet.append(list(row_values))
    else:
        output_book = load_workbook(output_path)
        output_sheet = output_book.active

    output_headers = {
        str(cell.value).strip().casefold(): index + 1
        for index, cell in enumerate(output_sheet[1])
        if cell.value is not None
    }
    for required_header in (
        "Applicant_ID",
        "Status",
        "Loan_risk_assistant_comment",
        "Reviewer Decision",
        "Risk Assessment",
    ):
        normalized_header = required_header.casefold()
        if normalized_header not in output_headers:
            column = output_sheet.max_column + 1
            output_sheet.cell(1, column, required_header)
            output_headers[normalized_header] = column

    output_id_column = output_headers["applicant_id"]
    output_rows: dict[str, int] = {}
    for row_index in range(2, output_sheet.max_row + 1):
        value = output_sheet.cell(row_index, output_id_column).value
        if value is None or not str(value).strip():
            continue
        applicant_id = str(value).strip().casefold()
        if applicant_id in output_rows:
            raise ValueError(f"Output workbook has duplicate applicant ID {value!r}")
        output_rows[applicant_id] = row_index

    for source_row in input_rows[1:]:
        if source_header_map["applicant_id"] >= len(source_row):
            continue
        applicant_id_value = source_row[source_header_map["applicant_id"]]
        if applicant_id_value is None or not str(applicant_id_value).strip():
            continue
        normalized_id = str(applicant_id_value).strip().casefold()
        output_row = output_rows.get(normalized_id)
        if output_row is None:
            output_row = output_sheet.max_row + 1
            output_rows[normalized_id] = output_row
            for column in range(1, output_sheet.max_column + 1):
                output_sheet.cell(output_row, column, None)

        for header, source_column in source_header_map.items():
            if header in protected_columns:
                continue
            output_column = output_headers.get(header)
            if output_column is None:
                output_column = output_sheet.max_column + 1
                output_sheet.cell(1, output_column, source_headers[source_column])
                output_headers[header] = output_column
            value = (
                source_row[source_column]
                if source_column < len(source_row)
                else None
            )
            output_sheet.cell(output_row, output_column, value)

    result_headers = (
        "Loan_risk_assistant_comment",
        "Reviewer Decision",
        "Risk Assessment",
    )
    existing_headers = [
        str(output_sheet.cell(1, column).value).strip()
        if output_sheet.cell(1, column).value is not None
        else ""
        for column in range(1, output_sheet.max_column + 1)
    ]
    normalized_headers = [header.casefold() for header in existing_headers if header]
    if len(normalized_headers) != len(set(normalized_headers)):
        raise ValueError("Output workbook contains duplicate column headers")
    result_header_names = {header.casefold() for header in result_headers}
    ordered_headers = [
        header
        for header in existing_headers
        if header and header.casefold() not in result_header_names
    ] + list(result_headers)
    if existing_headers != ordered_headers:
        data_rows = [
            {
                header.casefold(): output_sheet.cell(row_index, column).value
                for column, header in enumerate(existing_headers, start=1)
                if header
            }
            for row_index in range(2, output_sheet.max_row + 1)
        ]
        for row in output_sheet.iter_rows():
            for cell in row:
                cell.value = None
        for column, header in enumerate(ordered_headers, start=1):
            output_sheet.cell(1, column, header)
        for row_index, row_values in enumerate(data_rows, start=2):
            for column, header in enumerate(ordered_headers, start=1):
                output_sheet.cell(
                    row_index,
                    column,
                    row_values.get(header.casefold()),
                )

    output_book.save(output_path)
    output_book.close()


def calculate_emi(principal: float, annual_rate: float, months: int) -> float:
    monthly_rate = annual_rate / 12 / 100
    if monthly_rate == 0:
        return principal / months
    growth = (1 + monthly_rate) ** months
    return principal * monthly_rate * growth / (growth - 1)


def build_workflow(
    vector_store: Chroma,
    chat_model: ChatOllama,
    output_guardrail_agent,
):
    def load_applicant_node(state: WorkflowState) -> dict:
        workbook_path = Path(state["input_workbook_path"])
        application_path = Path(state["application_path"])
        identity = read_applicant_identity(workbook_path, state["applicant_id"])
        if not application_path.is_file():
            return {
                "applicant": None,
                "applicant_identity": identity,
                "application_missing": True,
                "input_errors": ["Application document is missing."],
            }

        try:
            applicant = read_applicant(
                workbook_path,
                application_path,
                state["applicant_id"],
            )
        except ValueError:
            return {
                "applicant": None,
                "applicant_identity": identity,
                "application_missing": False,
                "input_errors": ["Application data failed structural validation."],
            }
        return {
            "applicant": applicant.model_dump(),
            "applicant_identity": {
                "applicant_id": applicant.applicant_id,
                "full_name": applicant.full_name,
                "age": applicant.age,
            },
            "application_missing": False,
            "input_errors": [],
        }

    def input_guardrails_node(state: WorkflowState) -> dict:
        errors = validate_applicant_input(
            state.get("applicant"),
            state.get("input_errors", []),
        )
        print("\n--- Input guardrails ---")
        if errors:
            print("Input invalid:")
            for error in errors:
                print(f"- {error}")
        else:
            print("Input valid: required applicant fields and document statuses are structured.")
        return {"input_errors": errors, "input_valid": not errors}

    def route_input_guardrails(state: WorkflowState) -> str:
        return "valid" if state["input_valid"] else "invalid"

    def retrieve_policy_node(_state: WorkflowState) -> dict:
        if not _state.get("applicant_id"):
            raise ValueError("Policy retrieval requires an applicant ID")
        if _state.get("application_missing"):
            return {"policy_context": "", "source_pages": []}
        queries = (
            "personal loan eligibility age salaried registered company minimum income",
            "personal loan required documents submitted verified",
            "personal loan minimum maximum loan amount",
            "personal loan tenure interest rate EMI calculation formula",
        )
        context, pages = retrieve_policy_context(queries, vector_store, k=3)
        if not context:
            raise ValueError("Policy retrieval returned no relevant passages")

        return {"policy_context": context, "source_pages": pages}

    def assess_applicant_node(state: WorkflowState) -> dict:
        if state.get("application_missing"):
            narrative = AssessmentNarrative(
                summary="No application document was found. Assessment is pending.",
            )
            return {"narrative": narrative.model_dump()}

        applicant = Applicant.model_validate(state["applicant"])
        response = chat_model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a loan-officer assistant. Write exactly one short "
                        "factual sentence about the supplied applicant details only. "
                        "Do not infer protected traits, invent facts, assess or mention "
                        "eligibility, or make a lending recommendation or decision. "
                        "Do not add an introduction. Return plain text, not JSON."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Applicant facts:\n{applicant.model_dump_json(indent=2)}\n\n"
                        f"Retrieved policy:\n{state['policy_context']}"
                    )
                ),
            ]
        )
        if not isinstance(response.content, str):
            raise TypeError("The assessment model returned non-text content")
        summary = response.content.strip()
        if re.search(
            r"\b(eligible|eligibility|approved|approval|rejected|rejection)\b",
            summary,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "The assistant summary contained lending-decision language; "
                "review stopped rather than showing an unsafe summary."
            )
        narrative = AssessmentNarrative(summary=summary)
        return {"narrative": narrative.model_dump()}

    def guardrails_node(state: WorkflowState) -> dict:
        if state.get("application_missing"):
            pending_reason = (
                "No application document was found; eligibility and supporting "
                "documents cannot be assessed."
            )
            return {
                "checks": ["PENDING: " + pending_reason],
                "blockers": [pending_reason],
                "missing_documents": ["Application document"],
                "risk_indicators": [pending_reason],
                "recommendation": (
                    "Assessment pending; request the application document. "
                    "Approval without it will be marked HIGH RISK."
                ),
                "emi_calculated": None,
                "data_completeness_confidence": 0.0,
            }

        applicant = Applicant.model_validate(state["applicant"])
        checks: list[str] = []
        blockers: list[str] = []

        def check(condition: bool, passed: str, failed: str) -> None:
            checks.append(f"{'PASS' if condition else 'FAIL'}: {passed if condition else failed}")
            if not condition:
                blockers.append(failed)

        check(
            applicant.age > 21,
            f"Age is {applicant.age}, above the policy minimum of 21.",
            f"Age must be above 21; recorded age is {applicant.age}.",
        )
        check(
            applicant.employment_status.strip().casefold() == "salaried",
            "Employment status is Salaried.",
            f"Employment status must be Salaried; recorded value is {applicant.employment_status!r}.",
        )
        check(
            applicant.company_registered,
            "Employer is recorded as registered.",
            "Employer is not recorded as a registered company.",
        )
        check(
            applicant.monthly_income >= 30000,
            f"Monthly income is {applicant.monthly_income:,.2f}, meeting the 30,000 minimum.",
            f"Monthly income is below the 30,000 minimum: {applicant.monthly_income:,.2f}.",
        )

        missing_documents = [
            name
            for name in REQUIRED_DOCUMENTS
            if name not in applicant.documents
            or not applicant.documents[name].submitted
            or not applicant.documents[name].verified
        ]
        check(
            not missing_documents,
            "All 8 required documents are submitted and verified.",
            "Required documents missing or not verified: " + ", ".join(missing_documents),
        )
        check(
            100000 <= applicant.requested_amount <= 500000,
            f"Requested amount is within the 100,000-500,000 policy range ({applicant.requested_amount:,.2f}).",
            f"Requested amount is outside the 100,000-500,000 policy range ({applicant.requested_amount:,.2f}).",
        )

        expected_rate = TENURE_RATES.get(applicant.tenure_months)
        check(
            expected_rate is not None
            and math.isclose(
                applicant.annual_interest_rate,
                expected_rate,
                abs_tol=0.001,
            ),
            (
                f"Tenure {applicant.tenure_months} months uses the policy rate "
                f"of {expected_rate:.1f}%."
                if expected_rate is not None
                else f"Tenure {applicant.tenure_months} months is supported by policy."
            ),
            (
                f"Tenure {applicant.tenure_months} months is unsupported or its rate "
                f"does not match policy ({expected_rate}% expected; "
                f"{applicant.annual_interest_rate}% recorded)."
                if expected_rate is not None
                else f"Tenure {applicant.tenure_months} months is not listed in policy."
            ),
        )

        emi = calculate_emi(
            applicant.requested_amount,
            applicant.annual_interest_rate,
            applicant.tenure_months,
        )
        check(
            math.isclose(applicant.declared_emi, emi, abs_tol=1.0),
            f"Declared EMI {applicant.declared_emi:,.2f} matches the independently calculated {emi:,.2f}.",
            f"Declared EMI {applicant.declared_emi:,.2f} does not match the independently calculated {emi:,.2f}.",
        )
        check(
            applicant.applicant_confirmation,
            "Applicant confirmation is recorded.",
            "Applicant confirmation is missing.",
        )

        complete_documents = sum(
            name in applicant.documents
            and applicant.documents[name].submitted
            and applicant.documents[name].verified
            for name in REQUIRED_DOCUMENTS
        )
        completeness = round(
            (len(PROFILE_FIELDS) + complete_documents)
            / (len(PROFILE_FIELDS) + len(REQUIRED_DOCUMENTS)),
            2,
        )
        risk_indicators = (
            blockers
            if blockers
            else ["No deterministic policy-check risks identified."]
        )
        recommendation = (
            "All deterministic checks passed; send to a loan officer for review."
            if not blockers
            else "Policy checks failed; REJECT is recommended. Any approval override will be marked HIGH RISK."
        )
        return {
            "checks": checks,
            "blockers": blockers,
            "missing_documents": missing_documents,
            "risk_indicators": risk_indicators,
            "recommendation": recommendation,
            "emi_calculated": round(emi, 2),
            "data_completeness_confidence": completeness,
        }

    def output_guardrails_node(state: WorkflowState) -> dict:
        summary = AssessmentNarrative.model_validate(state["narrative"]).summary
        is_valid, errors = validate_output_with_agent(output_guardrail_agent, summary)
        print("\n--- Output guardrails ---")
        if errors:
            print("Output invalid:")
            for error in errors:
                print(f"- {error}")
        else:
            print("Output valid: assessment summary passed safety and PII checks.")
        return {
            "output_errors": errors,
            "output_valid": is_valid,
        }

    def route_output_guardrails(state: WorkflowState) -> str:
        if not state["output_valid"]:
            return "invalid"
        return "ready" if not state["blockers"] else "blocked"

    def prepare_ready_review(state: WorkflowState) -> dict:
        if state["blockers"]:
            raise RuntimeError("The ready-for-review branch received policy blockers")
        return {"approval_allowed": True}

    def prepare_blocked_review(state: WorkflowState) -> dict:
        if not state["blockers"]:
            raise RuntimeError("The blocked-review branch received no policy blockers")
        return {"approval_allowed": False}

    def reject_guardrail_failure_node(state: WorkflowState) -> dict:
        invalid_input = not state.get("input_valid", False)
        guardrail_name = "Input" if invalid_input else "Output"
        reason = (
            "Applicant data is missing or does not meet the required structure."
            if invalid_input
            else "AI assessment did not pass the required output safety checks."
        )
        summary = (
            "Application data failed input guardrails and was rejected before policy assessment."
            if invalid_input
            else "AI assessment failed output guardrails and was not presented for human review."
        )
        recommendation = (
            f"{guardrail_name} guardrail rejected the application. Correct the issue and resubmit."
        )
        checks = list(state.get("checks", []))
        checks.append(f"FAIL: {guardrail_name} guardrails: {reason}")
        blockers = list(state.get("blockers", []))
        blockers.append(reason)
        risk_indicators = list(state.get("risk_indicators", []))
        risk_indicators.append(reason)
        final_status = "REJECTED"
        decision = "REJECT"
        risk_assessment = "Low Risk"
        assistant_comment = "\n".join(
            (summary, recommendation, "Risk indicators: " + "; ".join(risk_indicators))
        )
        print(
            f"\n{guardrail_name} guardrail rejected this applicant before HITL. "
            "The rejection will be recorded in the output workbook."
        )
        return {
            "narrative": {"summary": summary},
            "checks": checks,
            "blockers": blockers,
            "missing_documents": state.get("missing_documents", []),
            "risk_indicators": risk_indicators,
            "recommendation": recommendation,
            "source_pages": state.get("source_pages", []),
            "data_completeness_confidence": state.get(
                "data_completeness_confidence", 0.0
            ),
            "reviewer_id": f"{guardrail_name} Guardrail",
            "decision": decision,
            "final_status": final_status,
            "risk_assessment": risk_assessment,
            "assistant_comment": assistant_comment,
        }

    def human_review_node(state: WorkflowState) -> dict:
        narrative = AssessmentNarrative.model_validate(state["narrative"])
        print("\n--- Loan application review ---")
        identity = state["applicant_identity"]
        print(f"Applicant: {identity['full_name']} ({identity['applicant_id']})")
        print(f"Summary: {narrative.summary}")
        print(f"Recommendation: {state['recommendation']}")
        print(f"Data completeness confidence: {state['data_completeness_confidence']:.0%}")
        if state["emi_calculated"] is not None:
            print(f"Independently calculated EMI: {state['emi_calculated']:,.2f}")
        if state["missing_documents"]:
            print(f"Missing/unverified documents: {', '.join(state['missing_documents'])}")
        print("\nPolicy checks:")
        for item in state["checks"]:
            print(f"- {item}")
        print("\nRisk indicators:")
        for item in state["risk_indicators"]:
            print(f"- {item}")
        if state["source_pages"]:
            print(f"\nPolicy source pages: {', '.join(map(str, state['source_pages']))}")
        print("\nThis is decision support only; the human reviewer makes the final decision.")

        if state["blockers"]:
            print(
                "\nWARNING: Approving despite failed or pending checks is an override "
                "and will be recorded as HIGH RISK."
            )
        choices = {"APPROVE", "REJECT", "REQUEST_INFO"}
        while True:
            decision = input(f"Reviewer decision {sorted(choices)}: ").strip().upper()
            if decision in choices:
                break
            print(f"Invalid decision. Choose one of: {', '.join(sorted(choices))}.")
        reviewer_id = input("Reviewer name or ID: ").strip()
        if not reviewer_id:
            raise ValueError("Reviewer name or ID cannot be blank")
        final_status = {
            "APPROVE": "APPROVED",
            "REJECT": "REJECTED",
            "REQUEST_INFO": "NEEDS_INFO",
        }[decision]
        risk_assessment = (
            "High Risk"
            if decision == "APPROVE" and state["blockers"]
            else "No Risk"
            if decision == "APPROVE"
            else "Low Risk"
        )
        assistant_comment = "\n".join(
            (
                narrative.summary,
                state["recommendation"],
                "Risk indicators: " + "; ".join(state["risk_indicators"]),
            )
        )
        return {
            "reviewer_id": reviewer_id,
            "decision": decision,
            "final_status": final_status,
            "risk_assessment": risk_assessment,
            "assistant_comment": assistant_comment,
        }

    def record_review_node(state: WorkflowState) -> dict:
        path = Path(state["history_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "applicant_id": state["applicant_id"],
            "reviewer_id": state["reviewer_id"],
            "decision": state["decision"],
            "final_status": state["final_status"],
            "risk_assessment": state["risk_assessment"],
            "assistant_comment": state["assistant_comment"],
            "policy_checks": state["checks"],
            "blocking_issues": state["blockers"],
            "missing_documents": state["missing_documents"],
            "risk_indicators": state["risk_indicators"],
            "input_guardrail_valid": state.get("input_valid", False),
            "recommendation": state["recommendation"],
            "summary": state["narrative"]["summary"],
            "source_pages": state["source_pages"],
            "input_guardrail_errors": state.get("input_errors", []),
            "output_guardrail_valid": state.get("output_valid", False),
            "output_guardrail_errors": state.get("output_errors", []),
            "data_completeness_confidence": state["data_completeness_confidence"],
        }
        with path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {}

    def update_workbook_node(state: WorkflowState) -> dict:
        path = Path(state["output_workbook_path"])
        workbook = load_workbook(path)
        sheet = workbook.active
        headers = {
            str(value.value).strip().casefold(): index + 1
            for index, value in enumerate(sheet[1])
            if value.value is not None
        }
        id_column = headers.get("applicant_id")
        status_column = headers.get("status")
        decision_column = headers.get("reviewer decision")
        risk_column = headers.get("risk assessment")
        comment_column = headers.get("loan_risk_assistant_comment")
        if any(
            column is None
            for column in (id_column, status_column, decision_column, risk_column, comment_column)
        ):
            raise ValueError(
                "Output workbook must contain Applicant_ID, Status, Reviewer Decision, "
                "Risk Assessment, and Loan_risk_assistant_comment columns"
            )
        matching_rows = [
            row_index
            for row_index in range(2, sheet.max_row + 1)
            if str(sheet.cell(row_index, id_column).value).strip().casefold()
            == state["applicant_id"].casefold()
        ]
        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one workbook row for {state['applicant_id']!r}, "
                f"found {len(matching_rows)}"
            )
        sheet.cell(matching_rows[0], status_column, state["final_status"])
        sheet.cell(matching_rows[0], decision_column, state["decision"])
        sheet.cell(matching_rows[0], risk_column, state["risk_assessment"])
        sheet.cell(matching_rows[0], comment_column, state["assistant_comment"])
        workbook.save(path)
        print(
            f"\nUpdated {path.name}: Decision: {state['decision']} "
            f"| Status: {state['final_status']} "
            f"| Risk Assessment: {state['risk_assessment']}"
        )
        return {}

    graph = StateGraph(WorkflowState)
    graph.add_node("load_applicant", load_applicant_node)
    graph.add_node("input_guardrails", input_guardrails_node)
    graph.add_node("retrieve_policy", retrieve_policy_node)
    graph.add_node("assess_applicant", assess_applicant_node)
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("output_guardrails", output_guardrails_node)
    graph.add_node("reject_guardrail_failure", reject_guardrail_failure_node)
    graph.add_node("prepare_ready_review", prepare_ready_review)
    graph.add_node("prepare_blocked_review", prepare_blocked_review)
    graph.add_node("human_review", human_review_node)
    graph.add_node("record_review", record_review_node)
    graph.add_node("update_workbook", update_workbook_node)
    graph.add_edge(START, "load_applicant")
    graph.add_edge("load_applicant", "input_guardrails")
    graph.add_conditional_edges(
        "input_guardrails",
        route_input_guardrails,
        {"valid": "retrieve_policy", "invalid": "reject_guardrail_failure"},
    )
    graph.add_edge("retrieve_policy", "assess_applicant")
    graph.add_edge("assess_applicant", "guardrails")
    graph.add_edge("guardrails", "output_guardrails")
    graph.add_conditional_edges(
        "output_guardrails",
        route_output_guardrails,
        {
            "ready": "prepare_ready_review",
            "blocked": "prepare_blocked_review",
            "invalid": "reject_guardrail_failure",
        },
    )
    graph.add_edge("prepare_ready_review", "human_review")
    graph.add_edge("prepare_blocked_review", "human_review")
    graph.add_edge("reject_guardrail_failure", "record_review")
    graph.add_edge("human_review", "record_review")
    graph.add_edge("record_review", "update_workbook")
    graph.add_edge("update_workbook", END)
    return graph.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a personal-loan application.")
    parser.add_argument(
        "--list-demo-applicants",
        action="store_true",
        help="List the applicant IDs reserved for demo data and exit.",
    )
    parser.add_argument(
        "--applicant-id",
        help="Review only this applicant. By default, review every applicant in the workbook.",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=SOURCE_WORKBOOK,
        help="Source applicant workbook (read-only; defaults to inputs/Input_Loan_Applicants.xlsx).",
    )
    parser.add_argument(
        "--output-workbook",
        type=Path,
        default=OUTPUT_WORKBOOK,
        help="Workbook for reviewer decisions and assistant assessments.",
    )
    parser.add_argument(
        "--application",
        type=Path,
        help="Application DOCX path (defaults to inputs/<applicant-id>_Loan_Application.docx).",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=INPUT_DIR / "review_history.jsonl",
        help="Append-only human review history path.",
    )
    args = parser.parse_args()
    if args.list_demo_applicants:
        print("Demo applicants: " + ", ".join(DEMO_APPLICANTS))
        return

    if args.application and not args.applicant_id:
        parser.error("--application can only be used together with --applicant-id")

    input_workbook_path = args.workbook.resolve()
    output_workbook_path = args.output_workbook.resolve()
    applicant_ids = (
        [args.applicant_id]
        if args.applicant_id
        else list_applicant_ids(input_workbook_path)
    )
    application_paths = {
        applicant_id: (
            args.application.resolve()
            if args.application
            else INPUT_DIR / f"{applicant_id}_Loan_Application.docx"
        )
        for applicant_id in applicant_ids
    }

    # Validate applicant rows before processing; application-data errors are handled
    # by the input-guardrail node so they can be recorded as rejected applications.
    for applicant_id in applicant_ids:
        read_applicant_identity(input_workbook_path, applicant_id)

    ensure_output_workbook(input_workbook_path, output_workbook_path)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    policy_pdf = find_policy_pdf()
    vector_store = load_vector_store(policy_pdf, embeddings)
    chat_model = ChatOllama(model=CHAT_MODEL, temperature=0)
    output_guardrail_agent = create_output_guardrail_agent()
    workflow = build_workflow(vector_store, chat_model, output_guardrail_agent)
    for index, applicant_id in enumerate(applicant_ids, start=1):
        print(f"\nReviewing applicant {index} of {len(applicant_ids)}")
        workflow.invoke(
            {
                "applicant_id": applicant_id,
                "input_workbook_path": str(input_workbook_path),
                "output_workbook_path": str(output_workbook_path),
                "application_path": str(application_paths[applicant_id].resolve()),
                "history_path": str(args.history.resolve()),
            }
        )


if __name__ == "__main__":
    main()
