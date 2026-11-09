import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure loan-approval-ai root is on sys.path for direct test discovery
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app


# ==============================================================================
# Test Fixtures & Sample Payloads
# ==============================================================================
def get_prime_applicant():
    """Return a prime borrower profile expected to pass underwriting."""
    return app.LoanApplication(
        applicant_name="Jane Doe",
        income=85000.0,
        credit_score=720.0,
        loan_amount=250000.0,
        years_employed=5.0,
    )


def get_subprime_applicant():
    """Return a subprime, high-debt borrower profile expected to be denied."""
    return app.LoanApplication(
        applicant_name="Marcus Vance",
        income=28000.0,
        credit_score=540.0,
        loan_amount=350000.0,
        years_employed=0.8,
    )


# ==============================================================================
# 1. Machine Learning Underwriting Tests
# ==============================================================================
def test_approved_applicant_prediction():
    """Verify high-credit, high-tenure applicant is Approved with high probability."""
    applicant = get_prime_applicant()
    response = app.process_loan(applicant)

    assert response["ml_decision"] == "Approved"
    assert response["approval_probability"] >= 0.90
    assert response["applicant_name"] == "Jane Doe"
    assert response["next_best_offer"] is None


def test_denied_applicant_prediction():
    """Verify subprime, high-debt applicant is Denied with low probability."""
    applicant = get_subprime_applicant()
    response = app.process_loan(applicant)

    assert response["ml_decision"] == "Denied"
    assert response["approval_probability"] <= 0.05
    assert response["applicant_name"] == "Marcus Vance"
    assert response["next_best_offer"] is not None


# ==============================================================================
# 2. Next-Best-Offer (NBO) Logic Tests
# ==============================================================================
def test_next_best_offer_for_approved_is_none():
    """Verify approved applicants receive null next_best_offer."""
    applicant = get_prime_applicant()
    response = app.process_loan(applicant)
    assert response["next_best_offer"] is None


def test_next_best_offer_structure_for_denied():
    """Verify denied applicant receives actionable recovery steps and sustainable principal."""
    applicant = get_subprime_applicant()
    response = app.process_loan(applicant)

    nbo = response["next_best_offer"]
    assert nbo is not None
    assert isinstance(nbo.actionable_steps, list)
    assert len(nbo.actionable_steps) >= 2
    assert nbo.suggested_loan_amount is not None
    # Suggested loan amount should be substantially lower than requested $350k and roughly 3x income
    assert nbo.suggested_loan_amount < applicant.loan_amount
    assert nbo.suggested_loan_amount == 84000.0


# ==============================================================================
# 3. Fair Lending & Guardrail Audit Tests
# ==============================================================================
def test_guardrails_word_boundary_safety():
    """Verify common words containing 'age' (e.g., 'encourage') do NOT trigger false demographic violations."""
    email_text = "Dear Applicant, we encourage you to reapply in 6 months as your credit profile evolves."
    status, notes = app.rule_based_compliance_check(email_text, ml_decision="Denied")

    assert "Passed" in status
    assert "Fair lending warning" not in notes


def test_guardrails_flags_prohibited_demographic_terms():
    """Verify explicit demographic terms (age, marital status, race) trigger Flagged compliance."""
    discriminatory_email = "Your loan was rejected due to your age and marital status."
    status, notes = app.rule_based_compliance_check(discriminatory_email, ml_decision="Denied")

    assert status == "Flagged"
    assert "age" in notes.lower() or "marital status" in notes.lower()


# ==============================================================================
# 4. Resilience & 503 Fallback Tests
# ==============================================================================
def test_resilience_to_gemini_503_spike(monkeypatch):
    """Verify endpoint gracefully falls back to local templates when Gemini returns 503 UNAVAILABLE."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "503 UNAVAILABLE: This model is currently experiencing high demand."
    )

    # Patch client getter to return mock client
    monkeypatch.setattr(app, "get_genai_client", lambda: mock_client)

    applicant = get_prime_applicant()
    response = app.process_loan(applicant)

    # API must still return successful result and not throw 500 / 502
    assert response["ml_decision"] == "Approved"
    assert response["approval_probability"] >= 0.90
    assert "Congratulations" in response["generated_notification_email"]
    assert "Passed" in response["compliance_status"]


# ==============================================================================
# 5. Database Persistence & Audit Trail Tests
# ==============================================================================
def test_database_persistence_and_retrieval():
    """Verify processed applications are permanently stored in SQLite and can be queried."""
    applicant = get_prime_applicant()
    response = app.process_loan(applicant)

    application_id = response["application_id"]
    assert isinstance(application_id, int)
    assert application_id > 0

    # Retrieve single record by ID
    record = app.get_audit_log_by_id(application_id)
    assert record["id"] == application_id
    assert record["applicant_name"] == "Jane Doe"
    assert record["ml_decision"] == "Approved"
    assert record["approval_probability"] == response["approval_probability"]


def test_database_stats_calculation():
    """Verify executive stats calculation aggregates totals and percentages accurately."""
    stats = app.get_audit_stats()

    assert "total_applications_audited" in stats
    assert "approved_count" in stats
    assert "denied_count" in stats
    assert "approval_rate_percent" in stats
    assert "compliance_pass_rate_percent" in stats

    assert stats["total_applications_audited"] == stats["approved_count"] + stats["denied_count"]
    assert 0.0 <= stats["approval_rate_percent"] <= 100.0
    assert 0.0 <= stats["compliance_pass_rate_percent"] <= 100.0
