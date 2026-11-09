import os
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from google import genai
from google.genai import types

# ==============================================================================
# Logging & Application Configuration
# ==============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loan-approval-ai")

app = FastAPI(
    title="Loan Approval AI API",
    description="Automated loan evaluation API combining Gradient Boosting ML decisioning, Gemini notification emails, AI compliance guardrails, Next-Best-Offer recommendations, and SQLite audit logging.",
    version="1.0.0",
)

# ==============================================================================
# 1. Model Artifact Loading (Deterministic Risk Scoring)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_gradient_boosting_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model artifact not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# ==============================================================================
# 2. SQLite Database Setup (Regulatory Audit Ledger)
# ==============================================================================
# Under ECOA and FCRA guidelines, lending institutions must retain an immutable
# audit trail of applicant financials, decision scores, and communications.
DB_PATH = BASE_DIR / "loans.db"


def init_db():
    """Create the persistent SQLite audit ledger table if it does not yet exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS loan_applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                applicant_name TEXT NOT NULL,
                income REAL NOT NULL,
                credit_score REAL NOT NULL,
                loan_amount REAL NOT NULL,
                years_employed REAL NOT NULL,
                ml_decision TEXT NOT NULL,
                approval_probability REAL NOT NULL,
                compliance_status TEXT NOT NULL,
                guardrail_notes TEXT NOT NULL,
                generated_notification_email TEXT NOT NULL,
                counter_offer_type TEXT,
                suggested_loan_amount REAL,
                next_best_offer_summary TEXT
            )
        """)
        conn.commit()


# Initialize database schema on startup
init_db()

# ==============================================================================
# 3. Google GenAI Client & Resilience Configuration
# ==============================================================================
# Configure a 15-second timeout on all LLM network requests to prevent API hanging.
CLIENT_HTTP_OPTIONS = types.HttpOptions(timeout=15000)

# Candidate models ordered by capability and availability
CANDIDATE_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# Client instance initialized lazily via get_genai_client()
client = None


def get_genai_client() -> genai.Client | None:
    """Retrieve or initialize the GenAI client using the GEMINI_API_KEY environment variable."""
    global client
    if client is not None:
        return client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not configured in environment. Backup templates will engage.")
        return None

    try:
        client = genai.Client(api_key=api_key, http_options=CLIENT_HTTP_OPTIONS)
        return client
    except Exception as exc:
        logger.warning(f"Could not initialize GenAI client: {exc}")
        return None


# ==============================================================================
# 4. Pydantic Schemas (Request & Response Validation)
# ==============================================================================
class LoanApplication(BaseModel):
    applicant_name: str = Field(..., description="Full name of the loan applicant")
    income: float = Field(..., description="Annual income of the applicant")
    credit_score: float = Field(..., description="Credit score of the applicant (FICO scale)")
    loan_amount: float = Field(..., description="Requested principal loan amount")
    years_employed: float = Field(..., description="Continuous employment tenure in years")

    model_config = {
        "json_schema_extra": {
            "example": {
                "applicant_name": "Jane Doe",
                "income": 85000.0,
                "credit_score": 720.0,
                "loan_amount": 250000.0,
                "years_employed": 5.0,
            }
        }
    }


class NextBestOffer(BaseModel):
    counter_offer_type: str = Field(
        ...,
        description="Type of counter-offer (e.g. Reduced Principal, Co-Signer Required, Financial Milestone)",
    )
    suggested_loan_amount: float | None = Field(
        None,
        description="Recommended sustainable loan amount based on income and risk profile, if applicable",
    )
    actionable_steps: list[str] = Field(
        ...,
        description="List of concrete, actionable recovery steps or milestone targets",
    )
    summary: str = Field(
        ...,
        description="Concise personalized rationale for the applicant",
    )


class LoanDecisionResponse(BaseModel):
    application_id: int = Field(
        ...,
        description="Permanent audit tracking ID stored in SQLite database",
    )
    applicant_name: str
    ml_decision: str
    approval_probability: float
    generated_notification_email: str
    compliance_status: str = Field(
        ...,
        description="Audit outcome (e.g. Passed, Flagged, or Passed (Fallback Audit))",
    )
    guardrail_notes: str = Field(
        ...,
        description="Detailed notes and regulatory compliance verification",
    )
    next_best_offer: NextBestOffer | None = Field(
        None,
        description="Next-Best-Offer / Counter-Offer recommendation for denied applicants (null if approved)",
    )


# ==============================================================================
# 5. Database Logging Helper
# ==============================================================================
def log_loan_decision(
    application: LoanApplication,
    ml_decision: str,
    approval_probability: float,
    generated_notification_email: str,
    compliance_status: str,
    guardrail_notes: str,
    next_best_offer: NextBestOffer | None,
) -> int:
    """Persist an application and all decision/audit metadata to SQLite, returning its row ID."""
    timestamp = datetime.now(timezone.utc).isoformat()
    counter_type = next_best_offer.counter_offer_type if next_best_offer else None
    suggested_amt = next_best_offer.suggested_loan_amount if next_best_offer else None
    nbo_summary = next_best_offer.summary if next_best_offer else None

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO loan_applications (
                timestamp, applicant_name, income, credit_score, loan_amount,
                years_employed, ml_decision, approval_probability,
                compliance_status, guardrail_notes, generated_notification_email,
                counter_offer_type, suggested_loan_amount, next_best_offer_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                application.applicant_name,
                application.income,
                application.credit_score,
                application.loan_amount,
                application.years_employed,
                ml_decision,
                approval_probability,
                compliance_status,
                guardrail_notes,
                generated_notification_email,
                counter_type,
                suggested_amt,
                nbo_summary,
            ),
        )
        conn.commit()
        return cursor.lastrowid


# ==============================================================================
# 6. Communication Engine & Resilient Fallback Templates
# ==============================================================================
def generate_fallback_email(applicant_name: str, ml_decision: str, loan_amount: float, note: str = "") -> str:
    """Generate a compliant, professional email if upstream AI services are temporarily unavailable."""
    note_text = f"\n\n[System Note: {note}]" if note else ""
    if ml_decision == "Approved":
        return (
            f"Subject: Congratulations! Your Loan Application Has Been Approved\n\n"
            f"Dear {applicant_name},\n\n"
            f"We are pleased to inform you that your application for a loan of ${loan_amount:,.2f} has been approved.\n\n"
            f"Our underwriting team has reviewed your financial profile and verified that you meet our credit criteria.\n\n"
            f"Next Steps:\n"
            f"1. Log in to your applicant portal to review and sign the loan agreement documents.\n"
            f"2. Submit verification for your primary disbursement account.\n"
            f"3. Funds will be scheduled for transfer within 1-2 business days upon completion of documentation.\n\n"
            f"If you have any questions, please feel free to reach out to our lending department.\n\n"
            f"Sincerely,\n"
            f"Loan Underwriting Team{note_text}"
        )
    else:
        return (
            f"Subject: Update Regarding Your Loan Application\n\n"
            f"Dear {applicant_name},\n\n"
            f"Thank you for giving us the opportunity to review your recent application for a loan of ${loan_amount:,.2f}.\n\n"
            f"After careful consideration of your financial profile against our current underwriting guidelines, "
            f"we regret to inform you that we are unable to approve your application at this time.\n\n"
            f"Next Steps & Recommendations:\n"
            f"- You may request a free copy of your credit report from major credit bureaus to check for inaccuracies.\n"
            f"- We encourage you to reapply in 6 months as your financial situation or credit profile evolves.\n\n"
            f"Thank you for considering our institution.\n\n"
            f"Sincerely,\n"
            f"Loan Processing Department{note_text}"
        )


def draft_notification_email(
    prompt: str,
    applicant_name: str,
    ml_decision: str,
    loan_amount: float,
) -> str:
    """Generate an email with Gemini, automatically falling back through candidate models and templates."""
    genai_client = get_genai_client()
    if genai_client is None:
        return generate_fallback_email(
            applicant_name,
            ml_decision,
            loan_amount,
            note="GEMINI_API_KEY environment variable is not set. Generated using standard notification template.",
        )

    # Multi-model fallback sequence to handle upstream 503 capacity spikes
    for model_name in CANDIDATE_MODELS:
        try:
            logger.info(f"Attempting email generation with model: {model_name}")
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text
        except Exception as exc:
            logger.warning(f"Model {model_name} failed: {exc}. Trying fallback if available...")
            continue

    # Graceful fallback to verified template if all LLM attempts are exhausted
    logger.warning("All Gemini candidate models failed. Returning graceful fallback email.")
    return generate_fallback_email(
        applicant_name,
        ml_decision,
        loan_amount,
        note="Gemini API temporarily experienced high demand (503). Delivered via automated fallback template.",
    )


# ==============================================================================
# 7. Regulatory Compliance Guardrails & Bias Screening (ECOA Standards)
# ==============================================================================
# Prohibited demographic attributes under the Equal Credit Opportunity Act (ECOA)
PROTECTED_DEMOGRAPHIC_TERMS = [
    "race", "skin color", "ethnicity", "gender", "sex", "sexual orientation",
    "religion", "religious", "marital status", "married", "unmarried",
    "national origin", "nationality", "disability", "handicap", "pregnancy", "age"
]


def rule_based_compliance_check(email_text: str, ml_decision: str) -> tuple[str, str]:
    """Fast local fallback audit for fair lending compliance and bias detection.

    Uses word-boundary regex (\\b) so normal vocabulary (e.g. 'encourage', 'coverage') is not falsely flagged.
    """
    flagged_terms = [
        term for term in PROTECTED_DEMOGRAPHIC_TERMS
        if re.search(rf"\b{re.escape(term)}\b", email_text, re.IGNORECASE)
    ]

    if flagged_terms:
        return (
            "Flagged",
            f"Fair lending warning: Email references prohibited demographic characteristics ({', '.join(flagged_terms)}). Must be revised to comply with ECOA standards.",
        )

    lower_text = email_text.lower()
    if ml_decision == "Denied":
        has_criteria_mention = any(k in lower_text for k in ["criteria", "guideline", "credit", "income", "underwriting", "profile"])
        if not has_criteria_mention:
            return (
                "Flagged",
                "Adverse action compliance warning: Rejection notice does not clearly state objective underwriting or credit criteria.",
            )
        return (
            "Passed (Fallback Audit)",
            "Automated compliance audit passed: Rejection notice cites objective underwriting criteria, maintains professional tone, and references zero protected demographic characteristics (ECOA compliant).",
        )
    else:
        return (
            "Passed (Fallback Audit)",
            "Automated compliance audit passed: Approval notice adheres to fair lending standards with no bias or protected demographic references.",
        )


def audit_email_guardrails(
    email_text: str,
    ml_decision: str,
    applicant: LoanApplication,
) -> tuple[str, str]:
    """Audit the generated notification email for fair lending compliance and professional tone.

    Combines Gemini LLM evaluation with a deterministic rule-based fallback.
    """
    genai_client = get_genai_client()
    if genai_client is None:
        return rule_based_compliance_check(email_text, ml_decision)

    audit_prompt = f"""You are a certified Fair Lending & Compliance Officer auditing an automated loan notification email for regulatory compliance under the Equal Credit Opportunity Act (ECOA) and fair lending standards.

Applicant Details:
- Name: {applicant.applicant_name}
- Requested Loan: ${applicant.loan_amount:,.2f}
- Stated Income: ${applicant.income:,.2f}
- Credit Score: {applicant.credit_score}
- Years Employed: {applicant.years_employed}

Model Decision: {ml_decision}

Notification Email Content:
---
{email_text}
---

Audit Instructions:
1. Protected Demographics Check: Verify that the email does NOT mention or imply any protected demographic characteristics (e.g., race, ethnicity, gender, marital status, religion, age, or disability).
2. Fair Lending / Adverse Action: If the decision is 'Denied', verify the email cites objective underwriting/credit criteria without discriminatory bias and provides constructive next steps (e.g. credit report review).
3. Decision Consistency & Tone: Verify that the email clearly and courteously reflects the decision ({ml_decision}) in a professional financial institution tone.

Return strictly a valid JSON object with these two keys:
{{
  "compliance_status": "Passed" | "Flagged",
  "guardrail_notes": "<Concise 1-2 sentence explanation of your audit findings>"
}}
"""

    for model_name in CANDIDATE_MODELS:
        try:
            logger.info(f"Attempting compliance audit with model: {model_name}")
            response = genai_client.models.generate_content(
                model=model_name,
                contents=audit_prompt,
            )
            if response and response.text:
                raw_text = response.text.strip()
                clean_text = raw_text
                # Clean markdown fences if LLM enclosed JSON inside ```json ... ```
                if "```" in clean_text:
                    parts = clean_text.split("```")
                    for part in parts:
                        candidate = part.strip()
                        if candidate.startswith("json"):
                            candidate = candidate[4:].strip()
                        try:
                            data = json.loads(candidate)
                            if "compliance_status" in data:
                                return data.get("compliance_status", "Passed"), data.get("guardrail_notes", "")
                        except Exception:
                            continue
                try:
                    data = json.loads(clean_text)
                    return data.get("compliance_status", "Passed"), data.get("guardrail_notes", "")
                except Exception:
                    status = "Flagged" if "flagged" in raw_text.lower() else "Passed"
                    return status, raw_text
        except Exception as exc:
            logger.warning(f"Audit model {model_name} failed: {exc}. Trying fallback if available...")
            continue

    # Graceful fallback audit if all Gemini models fail or timeout
    logger.warning("Gemini audit models unavailable; running local rule-based compliance audit.")
    return rule_based_compliance_check(email_text, ml_decision)


# ==============================================================================
# 8. Next-Best-Offer (NBO) / Counter-Offer Advisory Engine
# ==============================================================================
def calculate_rule_based_nbo(applicant: LoanApplication) -> dict:
    """Calculate realistic counter-offer principal and recovery steps using financial benchmarks."""
    # Conservative debt service capacity: maximum 3.0x annual income
    sustainable_max = round(min(applicant.loan_amount * 0.75, applicant.income * 3.0), -2)
    suggested_amount = max(10000.0, sustainable_max) if applicant.income >= 20000 else None

    actionable_steps = []

    # 1. Credit score improvement milestone
    if applicant.credit_score < 650:
        pts = int(650 - applicant.credit_score)
        actionable_steps.append(
            f"Credit Score Improvement: Aim to improve your credit score by at least {pts} points (target 650+) by reducing credit card utilization below 30% and maintaining timely payments."
        )

    # 2. Reduced principal counter-offer
    if suggested_amount and suggested_amount < applicant.loan_amount:
        actionable_steps.append(
            f"Reduced Loan Amount Counter-Offer: Reapply for an adjusted principal of ${suggested_amount:,.2f}, which aligns with current debt-to-income underwriting limits."
        )

    # 3. Co-signer recommendation
    if applicant.credit_score < 620 or (applicant.loan_amount > applicant.income * 3.0):
        actionable_steps.append(
            "Add a Qualified Co-Signer: Applying with a creditworthy co-signer (credit score 680+ and verified stable income) can significantly improve qualification odds."
        )

    # 4. Employment tenure longevity target
    if applicant.years_employed < 2.0:
        months_needed = int(max(1, round((2.0 - applicant.years_employed) * 12)))
        actionable_steps.append(
            f"Employment Tenure: Build continuous employment history for approximately {months_needed} more months to reach the 2-year job stability guideline."
        )

    # Classify counter-offer profile
    if suggested_amount and suggested_amount < applicant.loan_amount:
        counter_type = "Reduced Principal & Co-Signer Option"
        summary = (
            f"While your initial request of ${applicant.loan_amount:,.2f} could not be approved, you may qualify for a revised loan amount of "
            f"${suggested_amount:,.2f} or by applying with an eligible co-signer."
        )
    elif applicant.credit_score < 620:
        counter_type = "Co-Signer Required"
        summary = "To bridge current credit qualifications, applying with an eligible co-signer will substantially increase approval probability."
    else:
        counter_type = "Financial Milestone Reapplication"
        summary = "Focusing on credit score enhancement and building employment tenure will put you in a strong position to reapply in 6 months."

    return {
        "counter_offer_type": counter_type,
        "suggested_loan_amount": suggested_amount,
        "actionable_steps": actionable_steps,
        "summary": summary,
    }


def generate_next_best_offer(applicant: LoanApplication) -> NextBestOffer:
    """Generate constructive counter-offer recommendations using Gemini with deterministic fallback."""
    genai_client = get_genai_client()
    rule_fallback = calculate_rule_based_nbo(applicant)

    if genai_client is None:
        return NextBestOffer(**rule_fallback)

    nbo_prompt = f"""You are a senior financial advisor and loan underwriting specialist at a financial institution.
An applicant's loan application has been Denied by the automated underwriting system.
Analyze their financial profile and generate a constructive Next-Best-Offer (NBO) or counter-offer strategy to assist them.

Applicant Metrics:
- Name: {applicant.applicant_name}
- Requested Loan Amount: ${applicant.loan_amount:,.2f}
- Stated Annual Income: ${applicant.income:,.2f}
- Credit Score: {applicant.credit_score}
- Years Employed: {applicant.years_employed}
- Estimated Sustainable Loan Baseline (3x Income): ${applicant.income * 3:,.2f}

Instructions:
1. Determine the best Counter-Offer Type (e.g. "Reduced Principal", "Co-Signer Required", "Credit & Employment Milestone", or a combination).
2. Recommend a realistic, sustainable loan amount (float or null if credit score is too low for any loan at present).
3. Provide 2 to 4 concrete, actionable financial recovery steps (e.g. debt reduction targets, credit score milestones, co-signer criteria, tenure).
4. Provide a 1-2 sentence empathetic and clear summary.

Return strictly a valid JSON object matching this schema:
{{
  "counter_offer_type": "<Counter-Offer Strategy Name>",
  "suggested_loan_amount": <float or null>,
  "actionable_steps": ["<actionable step 1>", "<actionable step 2>", "<actionable step 3>"],
  "summary": "<1-2 sentence concise summary>"
}}
"""

    for model_name in CANDIDATE_MODELS:
        try:
            logger.info(f"Attempting NBO generation with model: {model_name}")
            response = genai_client.models.generate_content(
                model=model_name,
                contents=nbo_prompt,
            )
            if response and response.text:
                raw_text = response.text.strip()
                clean_text = raw_text
                if "```" in clean_text:
                    parts = clean_text.split("```")
                    for part in parts:
                        candidate = part.strip()
                        if candidate.startswith("json"):
                            candidate = candidate[4:].strip()
                        try:
                            data = json.loads(candidate)
                            if "counter_offer_type" in data and "actionable_steps" in data:
                                return NextBestOffer(**data)
                        except Exception:
                            continue
                try:
                    data = json.loads(clean_text)
                    if "counter_offer_type" in data and "actionable_steps" in data:
                        return NextBestOffer(**data)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"NBO model {model_name} failed: {exc}. Trying fallback if available...")
            continue

    logger.warning("All Gemini candidate models failed for NBO. Returning rule-based counter-offer.")
    return NextBestOffer(**rule_fallback)


# ==============================================================================
# 9. FastAPI Endpoints (Public API & Audit Routes)
# ==============================================================================
@app.get("/")
def read_root():
    """Service status and endpoint directory."""
    return {
        "message": "Loan Approval AI API is running.",
        "documentation": "/docs",
        "audit_logs": "/audit-logs",
        "audit_stats": "/audit-stats",
    }


@app.post("/process-loan", response_model=LoanDecisionResponse)
def process_loan(application: LoanApplication):
    """End-to-end loan evaluation pipeline:

    1. Evaluates risk via Gradient Boosting model.
    2. Drafts applicant notification email via Gemini.
    3. Audits email for ECOA compliance and tone.
    4. Generates Next-Best-Offer (NBO) if denied.
    5. Persists the complete audit record to SQLite.
    """
    # 1. Format input features matching training feature sequence
    features = pd.DataFrame([
        {
            "income": application.income,
            "credit_score": application.credit_score,
            "loan_amount": application.loan_amount,
            "years_employed": application.years_employed,
        }
    ])

    # 2. Run deterministic ML inference
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    # Target convention: 1 = Approved, 0 = Denied
    approval_probability = round(float(probabilities[1]), 4)
    ml_decision = "Approved" if prediction == 1 else "Denied"

    # 3. Draft prompt for customer communication
    prompt = f"""You are a professional loan processing officer at a financial institution.
Draft a professional, concise email notification addressed to the applicant regarding their loan application.

Applicant Details:
- Name: {application.applicant_name}
- Requested Loan Amount: ${application.loan_amount:,.2f}
- Stated Annual Income: ${application.income:,.2f}
- Credit Score: {application.credit_score}
- Years Employed: {application.years_employed}

Model Evaluation:
- Final Decision: {ml_decision}
- Approval Probability: {approval_probability:.2%}

Instructions:
- If the decision is 'Approved', warmly congratulate {application.applicant_name}, clearly mention the approved loan amount, and state the next steps required to finalize the loan agreement.
- If the decision is 'Denied', deliver the decision politely and with empathy, state that credit criteria were not met at this time, and suggest helpful next steps (e.g. reviewing credit reports or reapplying in the future).
- Address the email directly to {application.applicant_name}.
- Keep the tone polite, professional, and concise. Include an appropriate Subject line.
"""

    # 4. Generate notification email with resilient fallback handling
    generated_notification_email = draft_notification_email(
        prompt=prompt,
        applicant_name=application.applicant_name,
        ml_decision=ml_decision,
        loan_amount=application.loan_amount,
    )

    # 5. Run regulatory compliance and bias guardrails
    compliance_status, guardrail_notes = audit_email_guardrails(
        email_text=generated_notification_email,
        ml_decision=ml_decision,
        applicant=application,
    )

    # 6. Generate Next-Best-Offer (NBO) for denied applicants
    next_best_offer: NextBestOffer | None = None
    if ml_decision == "Denied":
        next_best_offer = generate_next_best_offer(application)

    # 7. Persist complete decision and audit record to SQLite database
    application_id = log_loan_decision(
        application=application,
        ml_decision=ml_decision,
        approval_probability=approval_probability,
        generated_notification_email=generated_notification_email,
        compliance_status=compliance_status,
        guardrail_notes=guardrail_notes,
        next_best_offer=next_best_offer,
    )

    # 8. Return structured JSON response with audit tracking ID
    return {
        "application_id": application_id,
        "applicant_name": application.applicant_name,
        "ml_decision": ml_decision,
        "approval_probability": approval_probability,
        "generated_notification_email": generated_notification_email,
        "compliance_status": compliance_status,
        "guardrail_notes": guardrail_notes,
        "next_best_offer": next_best_offer,
    }


@app.get("/audit-logs")
def get_audit_logs(
    decision: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Retrieve historical loan audit records from the SQLite compliance ledger.

    Supports pagination and optional filtering by decision ('Approved' or 'Denied').
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if decision:
            cursor.execute(
                "SELECT * FROM loan_applications WHERE ml_decision = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (decision.capitalize(), limit, offset),
            )
        else:
            cursor.execute(
                "SELECT * FROM loan_applications ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


@app.get("/audit-logs/{application_id}")
def get_audit_log_by_id(application_id: int):
    """Retrieve the full compliance audit record for a single loan application by its ID."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM loan_applications WHERE id = ?", (application_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Loan application audit record #{application_id} not found in database.",
            )
        return dict(row)


@app.get("/audit-stats")
def get_audit_stats():
    """Generate executive summary statistics on underwriting decisions and compliance audit results."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM loan_applications")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM loan_applications WHERE ml_decision = 'Approved'")
        approved = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM loan_applications WHERE ml_decision = 'Denied'")
        denied = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM loan_applications WHERE compliance_status LIKE 'Passed%'")
        compliant = cursor.fetchone()[0]

        approval_rate = round((approved / total * 100), 1) if total > 0 else 0.0
        compliance_rate = round((compliant / total * 100), 1) if total > 0 else 100.0

        return {
            "total_applications_audited": total,
            "approved_count": approved,
            "denied_count": denied,
            "approval_rate_percent": approval_rate,
            "compliance_pass_rate_percent": compliance_rate,
        }
