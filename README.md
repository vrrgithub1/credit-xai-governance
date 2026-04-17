# 🛡️ Project: Credit XAI – Ethical AI in Lending

**Objective: A Governance-First approach to Credit Scoring using Explainable AI (XAI) and Bias Mitigation.**

**Status:** Phase I (Data Ingestion & Framework Mappings)

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

## 📊 Model Performance & Compliance (NIST 'Measure' Phase)

To ensure the model is both performant and fair, we conducted a differential impact analysis. This verified that removing protected attributes (Age/Gender) did not compromise the model's ability to predict credit risk.

| Model Version | ROC-AUC Score | Variance | Status |
| :--- | :--- | :--- | :--- |
| Baseline (Full) | 0.8000 | - | Benchmark |
| **Baseline (Compliant)** | **0.7992** | **-0.0008** | ✅ **CERTIFIED** |

**Governance Insight:** The negligible decay (0.1%) proves that predictive signal is driven by objective financial behavior, not demographic shortcuts.

## 🏗️ Technical Architecture (Medallion Pattern)

To ensure data integrity and a clear audit trail, we utilize a Medallion Architecture:

1. **🟫 Bronze (Raw):** Ingestion of standard retail credit datasets (LendingClub/Statlog).

1. **🥈 Silver (Cleaned):** Feature engineering, handling of missing values, and categorical encoding.

1. **🥇 Gold (Inference):** The final XGBoost model integrated with the SHAP explainability layer.

## � Project Folder Structure

This repository is organized to support data ingestion, model training, explainability, and governance auditing.

* `data/` — Medallion storage with raw, cleaned, and derived datasets.
  * `bronze/` — Raw ingested credit data.
  * `silver/` — Cleaned and preprocessed training and test sets.
  * `gold/` — Final outputs for inference and model validation.
* `models/` — Saved model artifacts and serialized assets.
* `notebooks/` — Exploratory analysis and governance-focused investigations.
* `reports/` — Generated governance, fairness, and audit deliverables.
* `src/` — Core application logic, including data pipelines, model training, explainability, and bias auditing.

## �🛠️ The "Phase I" Stack

* **Modeling:** XGBoost (Classification)

* **Explainability:** SHAP

* **Fairness Audit:** Fairlearn

* **Data Pipeline:** Pandas / DuckDB

* **UI:** Streamlit (Interactive Loan Officer Dashboard)

**Status:** Phase IV (Bias Mitigation & Risk Treatment) - ✅ **COMPLIANT**

...

## 📈 Roadmap

* **[X] Phase I:** Data Ingestion, EDA, and NIST Mapping
* **[X] Phase II:** Baseline Model & Differential Impact Analysis ✅
* **[X] Phase III:** Explainability Layer (Local/Global SHAP)
* **[X] Phase IV:** Bias Auditing & Mitigation (Fairlearn) - **Achieved 0.9583 Fairness Ratio**
* **[✅] Phase V:** Launch Interactive Credit Dashboard (In Progress)

