from fastapi.testclient import TestClient

import backend_api

client = TestClient(backend_api.app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rejects_non_pdf_upload():
    response = client.post(
        "/api/decision",
        data={"claimQuery": "Check a fracture claim"},
        files={"policyPdf": ("policy.txt", b"not-pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Upload a PDF policy document."


def test_rejects_empty_query(monkeypatch):
    def _raise(*args, **kwargs):
        raise ValueError("Claim query is required.")

    monkeypatch.setattr(backend_api.decision_service, "evaluate_claim", _raise)
    response = client.post(
        "/api/decision",
        data={"claimQuery": "   "},
        files={"policyPdf": ("policy.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Claim query is required."


def test_returns_decision_payload(monkeypatch):
    expected = {
        "decision": "APPROVED",
        "decisionLabel": "ACCEPT",
        "query": "Test query",
        "documentName": "policy.pdf",
        "justification": "Covered under page 3.",
        "amount": 150000,
        "amountDisplay": "Rs 150,000",
        "sourcePages": [3],
        "confidence": 0.88,
        "timestamp": "2026-04-25T10:00:00",
        "ruleViolations": [],
        "sourceEvidence": [{"page": 3, "snippet": "Coverage clause"}],
        "missingFields": [],
        "policyRadar": {"highlights": ["Pre-existing: 24 months"]},
    }

    monkeypatch.setattr(
        backend_api.decision_service,
        "evaluate_claim",
        lambda file_bytes, filename, query: expected,
    )

    response = client.post(
        "/api/decision",
        data={"claimQuery": "Test query"},
        files={"policyPdf": ("policy.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == expected
