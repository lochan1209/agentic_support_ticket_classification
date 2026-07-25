import pytest
from fastapi.testclient import TestClient
from app import app, limiter

# Create TestClient
client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Fixture to reset slowapi rate limits before each test run."""
    limiter.reset()

def test_health_check():
    """1. Test that the /health endpoint returns a 200 status and healthy body."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_generate_text_success():
    """2. Test valid POST request to /v1/generate endpoint."""
    payload = {"prompt": "Tell me a story about AI"}
    response = client.post("/v1/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model"] == "mock-llm-core"
    assert "Processed payload prompt: 'Tell me a story about AI' successfully." in data["output"]

def test_generate_text_invalid_payload():
    """3. Test schema validation failure (missing required 'prompt' field)."""
    payload = {}  # Missing prompt
    response = client.post("/v1/generate", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_rate_limiting_exceeded():
    """4. Test that the 6th request triggers a 429 Too Many Requests error."""
    payload = {"prompt": "Test prompt"}
    
    # Send 5 allowed requests
    for _ in range(5):
        res = client.post("/v1/generate", json=payload)
        assert res.status_code == 200

    # The 6th request must trigger Rate Limit Exceeded
    rate_limited_response = client.post("/v1/generate", json=payload)
    assert rate_limited_response.status_code == 429


