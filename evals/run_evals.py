from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from backend.core.config import ROOT, Settings
from backend.services.ingestion import ArtifactStore, UploadPayload, chunk_document, parse_upload
from backend.services.retrieval import RetrievalService


async def main() -> int:
    cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))
    uploads = [
        UploadPayload(path.name, path.read_bytes())
        for path in sorted((ROOT / "data" / "synthetic-charts").glob("DOC-*.txt"))
    ]
    with tempfile.TemporaryDirectory(prefix="healthchat-eval-") as temporary:
        settings = Settings(artifact_dir=Path(temporary))
        artifacts = ArtifactStore(settings.artifact_dir)
        documents = [parse_upload(upload) for upload in uploads]
        chunks = [chunk for document in documents for chunk in chunk_document(document, settings)]
        artifacts.save(documents, chunks)
        retriever = RetrievalService(settings, artifacts)
        passed = 0
        for case in cases:
            result = await retriever.retrieve(case["query"])
            text = " ".join(chunk.content.lower() for chunk in result.chunks)
            roles = {chunk.source_role for chunk in result.chunks}
            patients = {chunk.patient_id for chunk in result.chunks if chunk.patient_id}
            chunk_ids = {chunk.chunk_id for chunk in result.chunks}
            ok = all(term in text for term in case["required_terms"])
            if case.get("expected_role"):
                ok = ok and roles == {case["expected_role"]}
            if case.get("patient_id"):
                ok = ok and patients <= {case["patient_id"]}
            if case.get("expect_empty"):
                ok = ok and not result.chunks
            if case.get("expected_chunk_ids"):
                ok = ok and set(case["expected_chunk_ids"]) <= chunk_ids
            passed += int(ok)
            print(
                f"{'PASS' if ok else 'FAIL'} {case['case_id']} "
                f"backend={result.backend} chunks={len(result.chunks)} "
                f"citations={sorted(chunk_ids)}"
            )
    print(f"{passed}/{len(cases)} deterministic retrieval evaluations passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
