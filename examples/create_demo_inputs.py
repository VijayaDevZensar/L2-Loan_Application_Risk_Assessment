from pathlib import Path

from docx import Document
from openpyxl import Workbook, load_workbook


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "inputs"
WORKBOOK_PATH = INPUT_DIR / "Input_Loan_Applicants.xlsx"
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
DEMO_APPLICANTS = (
    {
        "applicant_id": "APP001",
        "full_name": "Alex Sample",
        "age": 32,
        "employment_status": "Salaried",
        "company_registered": True,
        "monthly_income": 65000,
        "requested_amount": 250000,
        "tenure_months": 24,
        "annual_interest_rate": 12.5,
        "applicant_confirmation": True,
        "missing_documents": (),
    },
    {
        "applicant_id": "APP002",
        "full_name": "Blake Sample",
        "age": 29,
        "employment_status": "Salaried",
        "company_registered": True,
        "monthly_income": 55000,
        "requested_amount": 200000,
        "tenure_months": 24,
        "annual_interest_rate": 12.5,
        "applicant_confirmation": True,
        "missing_documents": ("Bank Statements", "Income Declaration"),
    },
    {
        "applicant_id": "APP003",
        "full_name": "Casey Sample",
        "age": 26,
        "employment_status": "Salaried",
        "company_registered": True,
        "monthly_income": 25000,
        "requested_amount": 150000,
        "tenure_months": 12,
        "annual_interest_rate": 12.0,
        "applicant_confirmation": True,
        "missing_documents": (),
    },
    {
        "applicant_id": "APP004",
        "full_name": "Drew Sample",
        "age": 35,
        "employment_status": "Salaried",
        "company_registered": True,
        "monthly_income": 72000,
        "requested_amount": 300000,
        "tenure_months": 36,
        "annual_interest_rate": 13.0,
        "applicant_confirmation": True,
        "missing_documents": (),
    },
    {
        "applicant_id": "APP005",
        "full_name": "Evan Sample",
        "age": 31,
        "application_document": False,
    },
)


def calculate_emi(principal: float, annual_rate: float, months: int) -> float:
    monthly_rate = annual_rate / 12 / 100
    growth = (1 + monthly_rate) ** months
    return principal * monthly_rate * growth / (growth - 1)


def update_applicant_workbook() -> None:
    if WORKBOOK_PATH.exists():
        workbook = load_workbook(WORKBOOK_PATH)
        sheet = workbook.active
        headers = {
            str(cell.value).strip().casefold(): index + 1
            for index, cell in enumerate(sheet[1])
            if cell.value is not None
        }
        required_headers = {"applicant_id", "full_name", "age", "status"}
        if not required_headers.issubset(headers):
            raise ValueError(
                "Existing workbook must contain Applicant_ID, Full_Name, Age, and Status columns"
            )
        if "risk assessment" not in headers:
            risk_column = sheet.max_column + 1
            sheet.cell(1, risk_column, "Risk Assessment")
            headers["risk assessment"] = risk_column
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Applicants"
        sheet.append(
            ["Applicant_ID", "Full_Name", "Age", "Status", "Risk Assessment"]
        )
        headers = {
            "applicant_id": 1,
            "full_name": 2,
            "age": 3,
            "status": 4,
            "risk assessment": 5,
        }

    headers_by_id: dict[str, list[int]] = {}
    for row_index in range(2, sheet.max_row + 1):
        value = sheet.cell(row_index, headers["applicant_id"]).value
        if value is not None:
            headers_by_id.setdefault(str(value).strip().casefold(), []).append(row_index)

    legacy_rows = headers_by_id.get("ap001", [])
    canonical_rows = headers_by_id.get("app001", [])
    if legacy_rows:
        if len(legacy_rows) > 1:
            raise ValueError("Legacy applicant ID AP001 appears more than once")
        legacy_row = legacy_rows[0]
        legacy_name = sheet.cell(legacy_row, headers["full_name"]).value
        if str(legacy_name).strip().casefold() == "alex sample":
            if canonical_rows:
                if len(canonical_rows) > 1:
                    raise ValueError("Applicant ID APP001 appears more than once")
                canonical_row = canonical_rows[0]
                old_status = sheet.cell(legacy_row, headers["status"]).value
                current_status = sheet.cell(canonical_row, headers["status"]).value
                if old_status and current_status and old_status != current_status:
                    raise ValueError(
                        "Legacy AP001 and APP001 have conflicting review statuses"
                    )
                if old_status and not current_status:
                    sheet.cell(canonical_row, headers["status"], old_status)
                sheet.delete_rows(legacy_row)
            else:
                sheet.cell(legacy_row, headers["applicant_id"], "APP001")

    rows_by_id: dict[str, list[int]] = {}
    for row_index in range(2, sheet.max_row + 1):
        value = sheet.cell(row_index, headers["applicant_id"]).value
        if value is not None:
            rows_by_id.setdefault(str(value).strip().casefold(), []).append(row_index)

    for applicant in DEMO_APPLICANTS:
        applicant_id = applicant["applicant_id"]
        rows = rows_by_id.get(applicant_id.casefold(), [])
        if len(rows) > 1:
            raise ValueError(f"Applicant ID {applicant_id} appears more than once in the workbook")
        if rows:
            row_index = rows[0]
        else:
            row_index = sheet.max_row + 1
            rows_by_id[applicant_id.casefold()] = [row_index]
            sheet.cell(row_index, headers["status"], "")

        sheet.cell(row_index, headers["applicant_id"], applicant_id)
        sheet.cell(row_index, headers["full_name"], applicant["full_name"])
        sheet.cell(row_index, headers["age"], applicant["age"])

    workbook.save(WORKBOOK_PATH)


def create_application_document(applicant: dict) -> None:
    if applicant.get("application_document") is False:
        return

    application = Document()
    application.add_heading("Synthetic Personal Loan Application", level=1)
    application.add_paragraph("DEMO DATA ONLY — not a real applicant.")
    application.add_paragraph(f"Applicant ID: {applicant['applicant_id']}")
    application.add_paragraph(f"Employment Status: {applicant['employment_status']}")
    application.add_paragraph(
        f"Company Registered: {'Yes' if applicant['company_registered'] else 'No'}"
    )
    application.add_paragraph(f"Monthly Income: {applicant['monthly_income']}")
    application.add_paragraph(f"Requested Loan Amount: {applicant['requested_amount']}")
    application.add_paragraph(f"Tenure Months: {applicant['tenure_months']}")
    application.add_paragraph(f"Annual Interest Rate: {applicant['annual_interest_rate']}")
    emi = calculate_emi(
        applicant["requested_amount"],
        applicant["annual_interest_rate"],
        applicant["tenure_months"],
    )
    application.add_paragraph(f"Declared EMI: {emi:.2f}")
    confirmation = "Yes" if applicant["applicant_confirmation"] else "No"
    application.add_paragraph(f"Applicant Confirmation: {confirmation}")
    application.add_heading("Document Status", level=2)
    missing_documents = set(applicant["missing_documents"])
    for name in REQUIRED_DOCUMENTS:
        submitted = "No" if name in missing_documents else "Yes"
        verified = "No" if name in missing_documents else "Yes"
        application.add_paragraph(
            f"DOCUMENT: {name} | Submitted: {submitted} | Verified: {verified}"
        )
    document_path = INPUT_DIR / f"{applicant['applicant_id']}_Loan_Application.docx"
    application.save(document_path)


def remove_renamed_app001_fixture() -> None:
    legacy_path = INPUT_DIR / "AP001_Loan_Application.docx"
    if not legacy_path.exists():
        return

    paragraphs = [paragraph.text.strip() for paragraph in Document(legacy_path).paragraphs]
    if (
        "DEMO DATA ONLY — not a real applicant." not in paragraphs
        or "Applicant ID: AP001" not in paragraphs
    ):
        raise ValueError(
            f"Refusing to remove non-demo or unexpected legacy document: {legacy_path}"
        )
    legacy_path.unlink()


def remove_missing_applicant_fixture() -> None:
    path = INPUT_DIR / "APP005_Loan_Application.docx"
    if not path.exists():
        return

    paragraphs = [paragraph.text.strip() for paragraph in Document(path).paragraphs]
    if (
        "DEMO DATA ONLY — not a real applicant." not in paragraphs
        or "Applicant ID: APP005" not in paragraphs
    ):
        raise ValueError(f"Refusing to remove non-demo or unexpected document: {path}")
    path.unlink()


def main() -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    update_applicant_workbook()
    for applicant in DEMO_APPLICANTS:
        create_application_document(applicant)
    remove_renamed_app001_fixture()
    remove_missing_applicant_fixture()
    print(f"Created/updated synthetic demo inputs in {INPUT_DIR}")
    print("APP001: complete application")
    print("APP002: missing Bank Statements and Income Declaration")
    print("APP003: salaried applicant with monthly income below 30,000")
    print("APP004: eligible salaried applicant with all required documents")
    print("APP005: no application document (assessment pending)")


if __name__ == "__main__":
    main()
