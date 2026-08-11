from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services.ingestion import UploadPayload

from .conftest import FakeEmbeddingService, FakeSearchIndex


def _client(settings, artifacts) -> TestClient:
    return TestClient(
        create_app(
            settings=settings,
            artifacts=artifacts,
            embeddings=FakeEmbeddingService(),
            search_admin=FakeSearchIndex(),
        )
    )


def test_query_contract_rejects_blank_malformed_and_oversized_input(settings, artifacts):
    with _client(settings, artifacts) as client:
        blank = client.post("/query", json={"query": "   "})
        malformed = client.post(
            "/query", content="{not-json", headers={"content-type": "application/json"}
        )
        oversized = client.post("/query", json={"query": "x" * 1001})

    for response in (blank, malformed, oversized):
        assert response.status_code == 422
        assert response.json()["category"] == "validation"
        assert response.headers["x-request-id"] == response.json()["request_id"]


def test_ingestion_rejects_missing_file_unsupported_extension_and_invalid_utf8(
    settings, artifacts
):
    with _client(settings, artifacts) as client:
        missing = client.post("/ingestion")
        unsupported = client.post(
            "/ingestion", files=[("files", ("note.csv", b"value", "text/csv"))]
        )
        invalid_utf8 = client.post(
            "/ingestion", files=[("files", ("note.txt", b"\xff\xfe", "text/plain"))]
        )

    assert missing.status_code == 422
    assert unsupported.status_code == 422
    assert "unsupported extension" in unsupported.json()["message"]
    assert invalid_utf8.status_code == 422
    assert "UTF-8" in invalid_utf8.json()["message"]


def test_ingestion_rejects_duplicate_document_ids_before_hosted_writes(
    settings, artifacts, curated_uploads
):
    embeddings = FakeEmbeddingService()
    search = FakeSearchIndex()
    app = create_app(
        settings=settings,
        artifacts=artifacts,
        embeddings=embeddings,
        search_admin=search,
    )
    upload = curated_uploads[0]

    with TestClient(app) as client:
        response = client.post(
            "/ingestion",
            files=[
                ("files", (upload.filename, upload.content, "text/plain")),
                ("files", ("duplicate.txt", upload.content, "text/plain")),
            ],
        )

    assert response.status_code == 422
    assert "duplicate DOCUMENT_ID" in response.json()["message"]
    assert embeddings.calls == 0
    assert search.documents == {}


def test_path_like_upload_filename_cannot_control_artifact_paths(
    settings, artifacts, curated_uploads
):
    upload = curated_uploads[0]
    payload = UploadPayload("../../outside.txt", upload.content, "text/plain")
    app = create_app(
        settings=settings,
        artifacts=artifacts,
        embeddings=FakeEmbeddingService(),
        search_admin=FakeSearchIndex(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/ingestion",
            files=[("files", (payload.filename, payload.content, payload.content_type))],
        )

    assert response.status_code == 200
    assert response.json()["files"][0]["document_id"] == "DOC-CLIN-P001-001"
    assert (artifacts.normalized_dir / "DOC-CLIN-P001-001.txt").exists()
    assert not (settings.artifact_dir.parent / "outside.txt").exists()


def test_unknown_routes_and_wrong_methods_are_bounded(settings, artifacts):
    with _client(settings, artifacts) as client:
        unknown = client.get("/not-a-route")
        wrong_method = client.get("/query")

    assert unknown.status_code == 404
    assert wrong_method.status_code == 405
    assert unknown.headers.get("x-request-id")
    assert wrong_method.headers.get("x-request-id")
