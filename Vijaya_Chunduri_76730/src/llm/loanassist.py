"""
LoanAssist — RAG-over-knowledge-base + Ollama LLM with safety guardrails.
Falls back to stub responses when Ollama is unavailable (for offline QE testing).
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
PROMPTS_DIR = ROOT / "prompts"

# Safety patterns that trigger immediate refusal before LLM call
INJECTION_PATTERNS = [
    r"ignore (all |previous |your )?instructions",
    r"forget (all |your )?previous",
    r"you are now",
    r"act as (an? )?(admin|developer|system|bank)",
    r"i am (a |an )?(fintrust |bank |system )?(admin|administrator|developer|auditor|regulator)",
    r"reveal.{0,30}(system prompt|instructions|password|secret|config)",
    r"show.{0,30}(system prompt|instructions)",
    r"disregard.{0,20}(rules|instructions)",
    r"override.{0,20}(rules|safety|instructions)",
    r"pretend.{0,20}you are",
    r"jailbreak",
    r"DAN mode",
    r"(fca|pra|regulatory).{0,30}(audit|override|order|mandate).{0,30}(provide|give|release|disclose)",
    r"(urgent|emergency).{0,30}(provide|give|release|disclose).{0,30}(data|database|customer|information)",
]

GUARANTEE_PATTERNS = [
    r"\bguarantee\b.{0,30}\b(loan|approval|approved|credit)\b",
    r"\bwill (definitely|certainly|surely)\b.{0,30}\b(approve|get|receive)\b",
    r"\bpromise\b.{0,30}\b(loan|approval)\b",
]

PII_REQUEST_PATTERNS = [
    r"(show|tell|give|reveal|access|get|find).{0,40}(other customer|another customer|customer'?s? (account|details|information|loan))",
    r"access.{0,30}(account|record|file).{0,30}(customer|user|person)",
    r"(account|loan).{0,30}(holder|owner|detail).{0,30}(for|of|about).{0,20}[A-Z][a-z]",
    r"details for (account holder|customer|user)",
]


def _load_knowledge() -> str:
    docs = []
    for f in sorted(KNOWLEDGE_DIR.glob("*.md")):
        docs.append(f"--- {f.name} ---\n" + f.read_text())
    return "\n\n".join(docs)


def _load_prompt(version: str = "v2") -> str:
    path = PROMPTS_DIR / f"loanassist_prompt_{version}.txt"
    if not path.exists():
        path = PROMPTS_DIR / "loanassist_prompt_v2.txt"
    return path.read_text()


def _simple_retrieve(question: str, knowledge: str, top_k: int = 3) -> list[str]:
    """Keyword-based retrieval — split knowledge into paragraphs and score by overlap."""
    paragraphs = [p.strip() for p in knowledge.split("\n\n") if len(p.strip()) > 50]
    question_words = set(re.sub(r"[^\w\s]", "", question.lower()).split())

    scored = []
    for para in paragraphs:
        para_words = set(re.sub(r"[^\w\s]", "", para.lower()).split())
        overlap = len(question_words & para_words)
        scored.append((overlap, para))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k] if _ > 0] or paragraphs[:2]


def _call_ollama(prompt: str, host: str = "http://localhost:11434", model: str = "llama3") -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "seed": 42},
        }
    ).encode()

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception:
        raise


def _stub_response(question: str, safety_flags: list[str]) -> str:
    """Deterministic fallback when Ollama is not available. Uses keyword routing."""
    if safety_flags:
        return (
            "I'm sorry, I cannot help with that request. "
            "Please contact a FinTrust adviser at 0800 000 0001 for assistance."
        )
    q_lower = question.lower()

    # Out-of-scope products — check BEFORE rate/fee patterns to avoid false matches
    if any(kw in q_lower for kw in ["mortgage", "savings account", "business loan", "commercial", "investment bond", "credit card", "insurance", "pension"]):
        return (
            "That product is outside the scope of LoanAssist. "
            "LoanAssist only covers FinTrust Bank personal loan products. "
            "For other products, please contact a FinTrust adviser at 0800 000 0001."
        )

    if any(kw in q_lower for kw in ["complaint", "unhappy", "dissatisfied", "escalat", "ombudsman", "fos", "how do i complain", "not happy"]):
        return (
            "If you are unhappy with any part of the process, please contact FinTrust Complaints: "
            "complaints@fintrust.example.com or 0800 000 0001. "
            "If unresolved within 8 weeks, you may escalate to the Financial Ombudsman Service (FOS). "
            "FinTrust Bank is authorised and regulated by the FCA."
        )

    if any(kw in q_lower for kw in ["product", "offer", "range", "loan type", "what loan", "which loan"]):
        return (
            "FinTrust Bank offers three personal loan products: "
            "(1) Standard Personal Loan: £5,000–£50,000, 12–84 months, indicative APR 6.9%–24.9%. "
            "(2) Premier Personal Loan: £50,001–£250,000 for Premier customers with income ≥ £75,000. "
            "(3) Home Improvement Loan: £2,000–£75,000 for renovation purposes. "
            "Actual rates depend on your credit profile and are confirmed after assessment."
        )
    if any(kw in q_lower for kw in ["rate", "interest", "apr", "percent"]):
        return (
            "FinTrust Bank's Standard Personal Loan has an indicative APR of 6.9% to 24.9%, "
            "depending on your credit profile and loan term. "
            "Your actual rate will be confirmed after a creditworthiness assessment."
        )
    if any(kw in q_lower for kw in ["fee", "charge", "cost", "early repayment", "repay early"]):
        return (
            "FinTrust Bank currently charges no arrangement fee. "
            "A late payment fee of £25 applies per missed payment. "
            "Early repayment charges are capped at 58 days' interest (statutory maximum)."
        )
    if any(kw in q_lower for kw in ["eligib", "qualify", "criteria", "can i get", "can i borrow", "am i eligible"]):
        return (
            "To be eligible for a FinTrust personal loan you must: "
            "(1) be a UK resident aged 18 or over, "
            "(2) have a regular income (employment, self-employment, or pension), "
            "(3) pass creditworthiness and affordability checks, "
            "(4) have no undischarged bankruptcy, and "
            "(5) hold or open a FinTrust current account. "
            "A formal assessment is required to confirm your individual eligibility."
        )
    if any(kw in q_lower for kw in ["document", "proof", "id", "passport", "payslip", "evidence"]):
        return (
            "Required documents typically include: "
            "valid photo ID (passport or driving licence), "
            "proof of address (utility bill or bank statement ≤ 3 months old), "
            "and proof of income (last 3 payslips or 2 years' self-assessment tax returns)."
        )
    if any(kw in q_lower for kw in ["repay", "direct debit", "monthly", "payment", "instalment", "borrow"]):
        return (
            "Repayments are made by monthly Direct Debit from your FinTrust current account. "
            "You may make additional lump-sum payments at any time. "
            "A repayment holiday of up to 2 months is available once per loan term, subject to approval."
        )
    if any(kw in q_lower for kw in ["who makes", "final decision", "who decides", "who approve", "decision on my loan", "how is the decision"]):
        return (
            "All lending decisions at FinTrust Bank are made by authorised bank staff. "
            "The AI triage model provides an advisory risk score only. "
            "A trained loan officer reviews the AI-assisted assessment before any final decision. "
            "Customers have the right to request a human review of any AI-assisted assessment."
        )

    if any(kw in q_lower for kw in ["apply", "process", "step", "how to", "application process", "how do i apply"]):
        return (
            "You can apply online at fintrust.example.com/loans or in branch. "
            "After a soft credit check, a full application triggers a credit search. "
            "A trained loan officer reviews the AI-assisted triage before any decision. "
            "Outcomes are usually provided within 2 business days."
        )
    return (
        "I don't have specific information on that topic in my approved knowledge base. "
        "Please contact a FinTrust adviser at 0800 000 0001 or visit fintrust.example.com for help."
    )


class LoanAssist:
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3"):
        self._host = ollama_host
        self._model = model
        self._knowledge = _load_knowledge()

    def _check_safety(self, message: str) -> list[str]:
        flags = []
        msg_lower = message.lower()
        for pat in INJECTION_PATTERNS:
            if re.search(pat, msg_lower):
                flags.append("prompt_injection")
                break
        for pat in GUARANTEE_PATTERNS:
            if re.search(pat, msg_lower):
                flags.append("guarantee_request")
                break
        for pat in PII_REQUEST_PATTERNS:
            if re.search(pat, msg_lower):
                flags.append("pii_request")
                break
        return flags

    def respond(self, message: str, prompt_version: str = "v2") -> dict:
        safety_flags = self._check_safety(message)

        if safety_flags:
            return {
                "reply": (
                    "I'm sorry, I cannot assist with that request. "
                    "If you need help with your loan application, please contact a FinTrust adviser at 0800 000 0001."
                ),
                "sources": [],
                "safety_flags": safety_flags,
            }

        retrieved = _simple_retrieve(message, self._knowledge)
        context = "\n\n".join(retrieved)

        prompt_template = _load_prompt(prompt_version)
        full_prompt = prompt_template.replace("{knowledge}", context) + f"\n\nCustomer question: {message}\n\nLoanAssist:"

        try:
            reply = _call_ollama(full_prompt, host=self._host, model=self._model)
            if not reply:
                reply = _stub_response(message, safety_flags)
                sources = ["stub_fallback"]
            else:
                sources = [Path(f).name for f in sorted(KNOWLEDGE_DIR.glob("*.md"))]
        except Exception:
            reply = _stub_response(message, safety_flags)
            sources = ["stub_fallback"]

        return {"reply": reply, "sources": sources, "safety_flags": []}
