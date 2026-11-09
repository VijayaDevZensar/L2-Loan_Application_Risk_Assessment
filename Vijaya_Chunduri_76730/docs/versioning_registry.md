# Model Registry and Versioning Record
## FinTrust Bank AI QE Capstone
**Version:** 1.0 | **Date:** 2026-08-21

---

## Why Reproducibility Matters in BFSI AI

In BFSI (Banking, Financial Services and Insurance), AI systems that influence lending decisions must be:
- **Auditable**: Regulators and compliance teams must be able to reconstruct exactly what model version produced a specific score and why.
- **Reproducible**: The same input + model version + feature pipeline must always produce the same output to support defect investigation and rollback.
- **Traceable**: Any change to data, features, model, or prompt must be traceable to a version, author, and test evidence.
- **Rollback-ready**: If a model regression is detected after deployment, the previous stable version must be restorable.

Failure to maintain versioning can result in regulatory findings (e.g., under SR 11-7 equivalent guidance), customer disputes, and inability to respond to fairness challenges.

---

## Version Manifest

| Artifact | Version | Date | Owner | Status | Notes |
|---|---|---|---|---|---|
| Dataset | loan_data_v1.0 | 2026-08-21 | Data Engineering | Active | 500 rows, synthetic, seeded SEED=42 |
| Schema | schema_v1.0 | 2026-08-21 | AI QE Team | Active | data/schemas/loan_schema_v1.json |
| Feature pipeline | pipeline_v1.0 | 2026-08-21 | AI QE Team | Active | models/pipeline_v1.pkl |
| ML model | risk_model_v1.0 | 2026-08-21 | AI QE Team | Candidate | models/risk_model_v1.pkl |
| Scaler | scaler_v1.0 | 2026-08-21 | AI QE Team | Active | models/scaler_v1.pkl |
| System prompt | loanassist_prompt_v1.0 | 2026-08-21 | AI QE Team | Superseded | prompts/loanassist_prompt_v1.txt |
| System prompt | loanassist_prompt_v2.0 | 2026-08-21 | AI QE Team | Active | prompts/loanassist_prompt_v2.txt |
| Knowledge base | kb_v1.0 | 2026-08-21 | Product Team | Active | knowledge/products.md, policy.md |
| Test suite | ai_qe_suite_v1.0 | 2026-08-21 | AI QE Team | Active | tests/ directory |

---

## Intentional Quality Gate Failure Demonstration

As required by the Word capstone completion criterion, an intentional failing gate was demonstrated:

**Scenario**: Monitoring rule DR-01 (mean_annual_income) was initially set to baseline 67,500 — incorrect value, causing spurious alerts on the clean production data.

**Steps:**
1. DR-01 triggered ALERT on normal production data (baseline too low)
2. Defect DFT-014 raised: "DR-01 monitoring rule fired on normal data (wrong training baseline)"
3. Baseline updated to actual training mean: £86,545
4. Re-run: DR-01 passes on normal scenario; only fires correctly on income_shift scenario
5. Re-registered in monitoring rules

This demonstrates the pipeline: **fail gate → identify root cause → fix → re-register → re-verify**.

---

## Model Registry Entry (risk_model_v1.0)

```json
{
  "version": "risk_model_v1.0",
  "algorithm": "LogisticRegression",
  "seed": 42,
  "status": "candidate",
  "metrics": {
    "accuracy": 0.84,
    "precision": 0.9275,
    "recall": 0.8533,
    "f1": 0.8889,
    "roc_auc": 0.9189
  },
  "thresholds": {
    "recall_min": 0.75,
    "f1_min": 0.72,
    "roc_auc_min": 0.80
  }
}
```

All metrics exceed minimum thresholds. Status remains "candidate" pending human Responsible AI sign-off.
