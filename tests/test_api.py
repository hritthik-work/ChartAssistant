from __future__ import annotations

from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.main import create_app

from .conftest import FakeEmbeddingService, FakeSearchIndex


def test_health_documents_and_static_ui(settings, artifacts):
    app = create_app(settings=settings, artifacts=artifacts)
    with TestClient(app) as client:
        health = client.get("/health")
        documents = client.get("/documents")
        index = client.get("/")
        favicon = client.get("/favicon.ico")
    assert health.status_code == 200
    assert health.json()["application_version"] == "1.0.0"
    assert health.json()["synthetic_only"] is True
    assert health.json()["prompt_version"] == "v5"
    assert app.version == "1.0.0"
    assert documents.json() == []
    assert index.status_code == 200
    assert "Chart Assistant" in index.text
    assert favicon.status_code == 200


def test_upload_ingestion_and_document_catalog(settings, artifacts, curated_uploads):
    search = FakeSearchIndex()
    app = create_app(
        settings=settings,
        artifacts=artifacts,
        embeddings=FakeEmbeddingService(),
        search_admin=search,
    )
    upload = curated_uploads[0]
    with TestClient(app) as client:
        response = client.post(
            "/ingestion",
            files=[("files", (upload.filename, upload.content, "text/plain"))],
        )
        documents = client.get("/documents")
    assert response.status_code == 200
    assert response.json()["documents_indexed"] == 1
    assert documents.status_code == 200
    assert documents.json()[0]["document_id"] == "DOC-CLIN-P001-001"


def test_ingestion_rejects_non_synthetic_file(settings, artifacts):
    app = create_app(
        settings=settings,
        artifacts=artifacts,
        embeddings=FakeEmbeddingService(),
        search_admin=FakeSearchIndex(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/ingestion",
            files=[("files", ("bad.txt", b"not synthetic", "text/plain"))],
        )
    assert response.status_code == 422
    assert response.json()["category"] == "validation"


def test_query_schema_rejects_extra_fields(settings, artifacts):
    app = create_app(settings=settings, artifacts=artifacts)
    with TestClient(app) as client:
        response = client.post("/query", json={"query": "valid query", "extra": True})
    assert response.status_code == 422
    assert response.json()["error"] == "request_error"


def test_safety_query_works_without_azure_configuration(settings, artifacts):
    app = create_app(settings=settings, artifacts=artifacts)
    with TestClient(app) as client:
        response = client.post("/query", json={"query": "How can I poison someone?"})
    assert response.status_code == 200
    assert response.json()["intent"] == "safety_refusal"
    assert response.json()["trace"]["model_calls"] == 0


def test_normal_query_without_chat_configuration_is_typed_503(settings, artifacts):
    app = create_app(settings=settings, artifacts=artifacts)
    with TestClient(app) as client:
        response = client.post("/query", json={"query": "What is documented for SYN-P001?"})
    assert response.status_code == 503
    assert response.json()["category"] == "configuration"
    assert response.headers["x-request-id"] == response.json()["request_id"]


def test_shallow_dependency_health_is_explicit(settings, artifacts):
    app = create_app(settings=settings, artifacts=artifacts)
    with TestClient(app) as client:
        aggregate = client.get("/health/services")
        chat = client.get("/health/azure-openai/chat")
        embeddings = client.get("/health/azure-openai/embeddings")
        search = client.get("/health/azure-search")
        search_query = client.get("/health/azure-search/query")
        search_ingestion = client.get("/health/azure-search/ingestion")
    assert aggregate.json()["status"] == "degraded"
    assert chat.json()["status"] == "not_configured"
    assert embeddings.json()["status"] == "not_configured"
    assert search.json()["status"] == "degraded"
    assert search.json()["operations"]["query"]["status"] == "not_configured"
    assert search.json()["operations"]["ingestion"]["status"] == "not_configured"
    assert search_query.json()["status"] == "not_configured"
    assert search_query.json()["credential"] == "query_key"
    assert search_ingestion.json()["status"] == "not_configured"
    assert search_ingestion.json()["credential"] == "admin_key"


def test_search_query_and_ingestion_configuration_are_independent(tmp_path):
    common = {
        "artifact_dir": tmp_path / "artifacts",
        "azure_search_endpoint": "https://unit-test.search.windows.net",
        "azure_search_index_name": "unit-test-index",
    }
    query_settings = Settings(
        **common,
        azure_search_query_key="query-key",
        azure_search_admin_key=None,
    )
    ingestion_settings = Settings(
        **common,
        azure_search_admin_key="admin-key",
        azure_search_query_key=None,
    )

    with TestClient(create_app(settings=query_settings)) as client:
        assert client.get("/health/azure-search/query").json()["status"] == "configured"
        assert client.get("/health/azure-search/ingestion").json()["status"] == "not_configured"

    with TestClient(create_app(settings=ingestion_settings)) as client:
        assert client.get("/health/azure-search/query").json()["status"] == "not_configured"
        assert client.get("/health/azure-search/ingestion").json()["status"] == "configured"
