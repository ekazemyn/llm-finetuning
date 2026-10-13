from fastapi.testclient import TestClient

from llm_finetuning.api import create_app


class FakeGenerator:
    backend = "fake"

    def generate(self, prompts, max_tokens, temperature):
        return [f"answer:{prompt}" for prompt in prompts]


def test_health_single_and_batch_generation(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with TestClient(create_app(FakeGenerator())) as client:
        assert client.get("/health").json() == {"status": "ready", "backend": "fake"}
        response = client.post("/generate", json={"prompt": "calculate"})
        assert response.status_code == 200
        assert response.json()["text"] == "answer:calculate"
        batch = client.post("/generate/batch", json={"prompts": ["a", "b"]})
        assert batch.json()["texts"] == ["answer:a", "answer:b"]


def test_api_key_is_optional_and_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    with TestClient(create_app(FakeGenerator())) as client:
        assert client.post("/generate", json={"prompt": "x"}).status_code == 401
        response = client.post(
            "/generate",
            json={"prompt": "x"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
