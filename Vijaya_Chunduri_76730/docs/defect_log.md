# Consolidated Defect Log
## FinTrust Bank AI QE Capstone — Days 1–3
**Version:** 1.0 | **Date:** 2026-08-21

| Defect ID | Day | Category | Title | Severity | Status | Evidence | Resolution |
|---|---|---|---|---|---|---|---|
| DFT-001 | 1 | Data | Duplicate application_ids in raw training dataset | HIGH | OPEN (raw only) | test_dq02_application_id_duplicate_detected_raw PASS | De-duplicate before training; clean dataset validated |
| DFT-002 | 1 | Data | Missing annual_income values in raw dataset (rows 5, 42, 87) | HIGH | OPEN (raw only) | test_dq03_missing_income_detected_raw PASS | Pipeline imputes median; clean dataset has zero nulls |
| DFT-003 | 1 | Data | Negative annual_income in raw dataset (row 20: -£15,000) | HIGH | OPEN (raw only) | test_dq04_negative_income_detected_raw PASS | Rejected in API validation; removed from clean data |
| DFT-004 | 1 | Data | Invalid age values (age=15 and age=150) in raw dataset | HIGH | OPEN (raw only) | test_dq05_invalid_age_detected_raw PASS | Age range [18–100] enforced in API and schema |
| DFT-005 | 1 | Data | Impossible credit scores (950 and -10) in raw dataset | HIGH | OPEN (raw only) | test_dq06_impossible_credit_score_detected_raw PASS | API rejects via Pydantic [300–850] constraint |
| DFT-006 | 1 | Data | Unsupported employment types ('Freelancer', 'Student') in raw dataset | MEDIUM | OPEN (raw only) | test_dq07_unsupported_employment_detected_raw PASS | API rejects unsupported categories |
| DFT-007 | 1 | Data | Extreme loan amount £9,999,999 in raw dataset | MEDIUM | OPEN (raw only) | test_dq14_extreme_loan_amount_detected_raw PASS | API loan_amount > 0 only; business rule review needed |
| DFT-008 | 1 | Data | Invalid loan_term_months (-1 sentinel) in raw dataset | MEDIUM | OPEN (raw only) | test_dq08_invalid_term_detected_raw PASS | API validates allowed terms {12,24,36,48,60,84,120} |
| DFT-009 | 1 | Data | Post-decision leakage field 'approval_flag' in raw training data | CRITICAL | MITIGATED | test_dq10_leakage_field_detected_in_raw PASS; test_dq10_no_leakage_field_in_clean PASS | Removed from clean pipeline; schema leakage_fields documented |
| DFT-010 | 1 | Model | 75% class imbalance in synthetic_target — high-risk dominates | MEDIUM | MITIGATED | test_dq13_class_imbalance_documented PASS | class_weight='balanced' applied; recall metric prioritised |
| DFT-011 | 2 | LLM | Stub LoanAssist returns incorrect topic response for keyword overlap | MEDIUM | FIXED | test_pt001 through pt015 all PASS | Keyword routing reordered; out-of-scope/complaint patterns checked first |
| DFT-012 | 2 | LLM | Mortgage/out-of-scope question incorrectly returned APR response | MEDIUM | FIXED | test_pt007 PASS | Out-of-scope products detected before rate keywords |
| DFT-013 | 2 | Pipeline | Single-row inference with missing age returned NaN after imputation | MEDIUM | FIXED | test_ml02, test_ml06 PASS | Fallback median dictionary added to handle single-row inference |
| DFT-014 | 3 | MLOps | DR-01 monitoring rule fired on normal data (wrong training baseline) | LOW | FIXED | drift_monitoring_normal.json updated | Baseline set to actual training mean £86,545 |
