# 🏦 Loan Approval AI - Intelligent Lending & Compliance Engine

A production-grade, multi-tier automated lending microservice combining deterministic Machine Learning risk scoring, Generative AI applicant notifications, agentic regulatory compliance guardrails, Next-Best-Offer (NBO) financial advisory, and SQLite audit logging.

## 💡 Situation & Problem Context

Imagine you walk into a traditional bank and apply for a loan. Normally, you have to wait a good amount of time while a loan officer reads your paperwork, runs your credit, writes you a letter, and has a compliance manager double-check that they didn’t break any anti-discrimination laws. And if you get rejected, they just hand you a cold "No" and you walk out unhappy.

This project creates an automated system that does that entire bank process in less than 1 second.

---

## 🎯 Architecture Overview

```
                                  +-----------------------------+
                                  |   Applicant JSON Payload    |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  | 1. Gradient Boosting ML     |
                                  |    (Deterministic Risk Math)|
                                  +--------------+--------------+
                                                 |
                                                 v
                       +-------------------------------------------------+
                       | 2. Google GenAI (Gemini Flash)                  |
                       |    - Drafts personalized notification email     |
                       |    - Automatic 503 fallback & timeout handling  |
                       +-------------------------+-----------------------+
                                                 |
                                                 v
                       +-------------------------------------------------+
                       | 3. Regulatory Guardrails & Compliance Audit     |
                       |    - ECOA fair-lending review                   |
                       |    - Word-boundary regex demographic filters    |
                       +-------------------------+-----------------------+
                                                 |
                                                 v
                       +-------------------------------------------------+
                       | 4. Next-Best-Offer (NBO) Advisory Engine        |
                       |    - Generates counter-offers for denials       |
                       |    - Debt-to-income benchmarks & milestones     |
                       +-------------------------+-----------------------+
                                                 |
                                                 v
                       +-------------------------------------------------+
                       | 5. SQLite Regulatory Audit Ledger               |
                       |    - Immutable persistence & tracking IDs       |
                       |    - Live compliance analytics & stats          |
                       +-------------------------------------------------+
```

---

## 🚀 Key Features

* **Deterministic ML Underwriting:** Scikit-learn Gradient Boosting Classifier (`loan_gradient_boosting_model.pkl`) evaluates credit metrics (`income`, `credit_score`, `loan_amount`, `years_employed`) with zero mathematical hallucination.
* **Resilient Generative Communications:** Google GenAI SDK (`gemini-2.5-flash` $\rightarrow$ `gemini-2.0-flash` $\rightarrow$ `gemini-1.5-flash`) drafts custom acceptance and adverse action letters with sub-second fallback protection against upstream 503 cloud spikes.
* **Fair Lending Guardrails (ECOA Compliance):** Audits generated text for fair lending standards, utilizing regex word boundaries (`\b`) to eliminate false positives on common words (e.g., *"encourage"*) while strictly flagging prohibited demographic terms.
* **Next-Best-Offer (NBO) Logic:** Converts adverse action rejections into constructive customer recovery pathways by calculating sustainable debt capacities (e.g. offering an \$84k counter-offer on a \$350k request, adding co-signers, and target milestones).
* **Compliance Audit Trail:** Every application, decision, probability score, and email is permanently logged to an SQLite database with auto-incrementing tracking IDs and executive analytics.
* **100% Automated Test Coverage:** Comprehensive `pytest` suite testing ML inference, guardrail edge cases, 503 resilience, and database persistence.
* **Zero-Setup Containerization:** Fully containerized with `Dockerfile` and `docker-compose.yml` for instant, one-command deployment on any operating system.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI, Uvicorn, Pydantic v2
* **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
* **Generative AI:** Google GenAI SDK (Gemini Flash)
* **Database & Persistence:** SQLite3
* **Testing:** Pytest
* **Containerization & DevOps:** Docker, Docker Compose

---

## ⚡ Quick Start

### Option A: Run via Docker Compose (Recommended - Zero Setup)
```bash
# Clone the repository and navigate into the project directory
cd loan-approval-ai

# Start the container
docker compose up --build
```
The API and interactive Swagger documentation will be immediately accessible at:
* **Interactive Documentation (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Audit Ledger:** [http://127.0.0.1:8000/audit-logs](http://127.0.0.1:8000/audit-logs)
* **Executive Stats:** [http://127.0.0.1:8000/audit-stats](http://127.0.0.1:8000/audit-stats)

---

### Option B: Run Locally with Python

```bash
# 1. Navigate to the project directory
cd loan-approval-ai

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key (Optional; fallback templates engage automatically if omitted)
export GEMINI_API_KEY="your-api-key-here"  # On Windows PowerShell: $env:GEMINI_API_KEY="your-key"

# 5. Start the FastAPI server
uvicorn app:app --reload
```

---

## 🧪 Running the Automated Test Suite

### Run via Docker:
```bash
docker compose run loan-approval-api pytest -v
```

### Run Locally:
```bash
pytest -v
# OR using the built-in runner:
python run_tests.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/process-loan` | Runs ML evaluation, drafts Gemini email, audits compliance, generates NBO, and persists record. |
| `GET` | `/audit-logs` | Retrieves paginated historical loan records (optional query filter: `?decision=Approved` or `?decision=Denied`). |
| `GET` | `/audit-logs/{id}` | Fetches full regulatory audit file for a single application ID. |
| `GET` | `/audit-stats` | Calculates live underwriting KPIs: total audited, approval rate %, and compliance rate %. |
| `GET` | `/docs` | Interactive Swagger UI for live browser testing. |

---

## 📋 Sample Request & Response

### Request (`POST /process-loan`):
```json
{
  "applicant_name": "Marcus Vance",
  "income": 28000.0,
  "credit_score": 540.0,
  "loan_amount": 350000.0,
  "years_employed": 0.8
}
```

### Response:
```json
{
  "application_id": 1,
  "applicant_name": "Marcus Vance",
  "ml_decision": "Denied",
  "approval_probability": 0.0002,
  "generated_notification_email": "Subject: Update Regarding Your Loan Application\n\nDear Marcus Vance,\n\nThank you for giving us the opportunity to review your recent application for a loan of $350,000.00...",
  "compliance_status": "Passed (Fallback Audit)",
  "guardrail_notes": "Automated compliance audit passed: Rejection notice cites objective underwriting criteria, maintains professional tone, and references zero protected demographic characteristics (ECOA compliant).",
  "next_best_offer": {
    "counter_offer_type": "Reduced Principal & Co-Signer Option",
    "suggested_loan_amount": 84000.0,
    "actionable_steps": [
      "Credit Score Improvement: Aim to improve your credit score by at least 110 points (target 650+) by reducing credit card utilization below 30% and maintaining timely payments.",
      "Reduced Loan Amount Counter-Offer: Reapply for an adjusted principal of $84,000.00, which aligns with current debt-to-income underwriting limits.",
      "Add a Qualified Co-Signer: Applying with a creditworthy co-signer (credit score 680+ and verified stable income) can significantly improve qualification odds.",
      "Employment Tenure: Build continuous employment history for approximately 14 more months to reach the 2-year job stability guideline."
    ],
    "summary": "While your initial request of $350,000.00 could not be approved, you may qualify for a revised loan amount of $84,000.00 or by applying with an eligible co-signer."
  }
}
```

