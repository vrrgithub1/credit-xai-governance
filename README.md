# 🛡️ Project: Credit XAI – Ethical AI in Lending

**Objective: A Governance-First approach to Credit Scoring using Explainable AI (XAI) and Bias Mitigation.**

**Status:** Phase I (Data Ingestion & Framework Mapping)

**Frameworks:** NIST AI RMF | ISO/IEC 42001 | 4/5ths Rule for Fairness

## 📖 Project Charter

In traditional credit risk, "Black Box" models are a liability. Under the EU AI Act and NIST guidelines, high-stakes models like credit scoring must be Transparent, Explainable, and Fair.

This project implements a modernized credit decisioning engine that moves beyond simple "Approve/Deny" outcomes by providing game-theoretic reason codes for every decision.

## 🏛️ Governance & NIST Alignment

We map our development lifecycle directly to the **NIST AI RMF pillars**:

| NIST Pillar | Implementation Strategy |
| :--- | :--- |
| Govern | 21+ years of institutional oversight applied to feature selection and model lifecycle. |
| Map | Identification of "Proxies for Bias" during the EDA phase to prevent indirect discrimination. |
| Measure | Quantifying fairness via Disparate Impact Ratios and performance via AUC-ROC. |
| Manage | Implementing SHAP (Shapley Additive Explanations) to provide human-readable "Reason Codes." |

## 🏗️ Technical Architecture (Medallion Pattern)

To ensure data integrity and a clear audit trail, we utilize a Medallion Architecture:

1. **🟫 Bronze (Raw):** Ingestion of standard retail credit datasets (LendingClub/Statlog).

1. **🥈 Silver (Cleaned):** Feature engineering, handling of missing values, and categorical encoding.

1. **🥇 Gold (Inference):** The final XGBoost model integrated with the SHAP explainability layer.

## 🛠️ The "Phase I" Stack

* **Modeling:** XGBoost (Classification)

* **Explainability:** SHAP

* **Fairness Audit:** Fairlearn

* **Data Pipeline:** Pandas / DuckDB

* **UI:** Streamlit (Interactive Loan Officer Dashboard)

## 📈 Roadmap

* **[ ] Phase I:** Data Ingestion, EDA, and NIST Mapping (Current)

* **[ ] Phase II:** Baseline Model & Feature Importance

* **[ ] Phase III:** Explainability Layer (Local/Global SHAP)

* **[ ] Phase IV:** Bias Auditing & Mitigation (Fairlearn)

* **[ ] Phase V:** Launch Interactive Credit Dashboard
