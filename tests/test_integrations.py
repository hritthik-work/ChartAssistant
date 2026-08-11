from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.core.config import Settings
from backend.models import ModelCandidate
from backend.services.agent import MafAgentProvider
from backend.services.azure import AzureSearchService


def configured_settings(tmp_path):
    return Settings(
        artifact_dir=tmp_path / "artifacts",
        azure_openai_endpoint="https://unit-test.openai.azure.com/",
        azure_openai_api_key="unit-test-key",
        azure_openai_chat_deployment="unit-test-chat",
        azure_openai_embedding_deployment="unit-test-embedding",
        azure_search_endpoint="https://unit-test.search.windows.net",
        azure_search_admin_key="unit-test-admin",
        azure_search_query_key="unit-test-query",
        azure_search_index_name="unit-test-index",
    )


def test_maf_provider_builds_with_explicit_azure_configuration(tmp_path):
    provider = MafAgentProvider(configured_settings(tmp_path))
    assert provider.client.model == "unit-test-chat"
    assert provider.prompt_version == "v5"
    assert "## 7. Final validation" in provider.system_prompt


def test_azure_search_schema_contains_vector_and_semantic_configuration(tmp_path):
    service = AzureSearchService(configured_settings(tmp_path), require_admin=True)
    schema = service._schema(3)
    fields = {field.name: field for field in schema.fields}
    assert fields["content_vector"].vector_search_dimensions == 3
    assert schema.semantic_search.configurations[0].name == "healthchat-semantic"


def test_incompatible_search_schema_is_identified_before_hash_query(tmp_path):
    service = AzureSearchService(configured_settings(tmp_path), require_admin=True)
    legacy_index = SimpleNamespace(
        fields=[
            SimpleNamespace(name=name)
            for name in ("chunk_id", "document_id", "title", "text", "embedding")
        ]
    )

    detail = service._schema_mismatch_detail(legacy_index)

    assert detail is not None
    assert "content_hash" in detail
    assert "content_vector" in detail


@pytest.mark.asyncio
async def test_maf_does_not_force_unsupported_temperature(tmp_path):
    provider = MafAgentProvider(configured_settings(tmp_path))
    captured: dict = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["default_options"] = kwargs["default_options"]

        async def run(self, _query, *, options):
            captured["run_options"] = options
            return SimpleNamespace(
                value=ModelCandidate(
                    status="refused",
                    intent="out_of_scope",
                    answer="Not in scope.",
                    citations=[],
                    missing_information=[],
                ),
                usage_details=None,
            )

    provider.agent_class = FakeAgent
    await provider.generate("test query", object(), object())

    assert "temperature" not in captured["default_options"]
    assert "temperature" not in captured["run_options"]
