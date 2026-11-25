import math

from langchain.agents import create_agent
from langchain.agents.middleware import PIIDetectionError, PIIMiddleware
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:latest"
REQUIRED_APPLICANT_FIELDS = (
    "applicant_id",
    "full_name",
    "age",
    "employment_status",
    "company_registered",
    "monthly_income",
    "requested_amount",
    "tenure_months",
    "annual_interest_rate",
    "declared_emi",
    "applicant_confirmation",
    "documents",
)

INPUT_PII_MIDDLEWARE = (
    PIIMiddleware("email", strategy="redact", apply_to_input=True),
    PIIMiddleware("credit_card", strategy="redact", apply_to_input=True),
    PIIMiddleware(
        "phone",
        detector=r"(?<!\w)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\w)",
        strategy="redact",
        apply_to_input=True,
    ),
    PIIMiddleware(
        "aadhaar",
        detector=r"(?<!\d)(?:\d{4}[\s-]?){2}\d{4}(?!\d)",
        strategy="redact",
        apply_to_input=True,
    ),
    PIIMiddleware(
        "pan",
        detector=r"(?i)\b[A-Z]{5}\d{4}[A-Z]\b",
        strategy="redact",
        apply_to_input=True,
    ),
    PIIMiddleware(
        "secret",
        detector=r"\b(?:sk|api)[-_][A-Za-z0-9_-]{20,}\b",
        strategy="block",
        apply_to_input=True,
    ),
)


def validate_applicant_input(
    applicant: dict | None,
    parse_errors: list[str] | None = None,
) -> list[str]:
    errors = list(parse_errors or [])
    if applicant is None:
        if not errors:
            errors.append("Applicant application data is unavailable.")
        return errors

    for field in REQUIRED_APPLICANT_FIELDS:
        if field not in applicant or applicant[field] is None:
            errors.append(f"Required applicant field is missing: {field}.")

    for field in ("applicant_id", "full_name", "employment_status"):
        value = applicant.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"Applicant field {field} must be a non-empty string.")

    for field in ("company_registered", "applicant_confirmation"):
        value = applicant.get(field)
        if value is not None and not isinstance(value, bool):
            errors.append(f"Applicant field {field} must be a boolean.")

    documents = applicant.get("documents")
    if documents is not None and not isinstance(documents, dict):
        errors.append("Applicant documents must be a mapping of document statuses.")
    elif isinstance(documents, dict):
        for name, status in documents.items():
            if not isinstance(status, dict):
                errors.append(f"Document status for {name!r} must be a mapping.")
                continue
            if not isinstance(status.get("submitted"), bool):
                errors.append(f"Document {name!r} has an invalid submitted status.")
            if not isinstance(status.get("verified"), bool):
                errors.append(f"Document {name!r} has an invalid verified status.")

    age = applicant.get("age")
    if age is not None and (not isinstance(age, int) or isinstance(age, bool) or age <= 0):
        errors.append("Applicant age must be a positive whole number.")
    for field in (
        "monthly_income",
        "requested_amount",
        "annual_interest_rate",
        "declared_emi",
    ):
        value = applicant.get(field)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            errors.append(f"Applicant field {field} must be a positive number.")

    tenure = applicant.get("tenure_months")
    if tenure is not None and (
        not isinstance(tenure, int) or isinstance(tenure, bool) or tenure <= 0
    ):
        errors.append("Applicant tenure must be a positive whole number.")

    return errors


def create_input_guardrail_agent():
    return create_agent(
        model=ChatOllama(model=MODEL_NAME, temperature=0),
        tools=[],
        system_prompt=(
            "You are a safe loan-application information assistant. Give general "
            "guidance about application documents and process only. Do not approve "
            "or reject a loan, make eligibility decisions, or infer protected "
            "characteristics. Personal identifiers in the user message may have "
            "been redacted; never try to reconstruct them. Ask the user not to "
            "share passwords, API keys, or other authentication secrets."
        ),
        middleware=INPUT_PII_MIDDLEWARE,
    )


def main() -> None:
    agent = create_input_guardrail_agent()
    print("Loan application input guardrail")
    print("Enter a question, or type 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\nGoodbye.")
            return
        if user_input.casefold() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not user_input:
            continue

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
        except PIIDetectionError:
            print(
                "Input blocked: an API-key-like secret was detected. "
                "Remove it and retry.\n"
            )
            continue
        print(f"\nAssistant: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
