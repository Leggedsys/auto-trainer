from __future__ import annotations

from pathlib import Path

from field_discovery import (
    build_field_discovery_prompt,
    discover_fields,
    load_source_snippet,
)


class StubLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_build_field_discovery_prompt_includes_source_snippets() -> None:
    prompt = build_field_discovery_prompt(
        task_id="Template-Autotrain-v0",
        task_type="ManagerBasedRLEnv",
        source_snippets=[
            {
                "label": "agent cfg",
                "path": "/tmp/agent.py",
                "content": "learning_rate=1.0e-3",
            }
        ],
    )

    assert "Template-Autotrain-v0" in prompt
    assert "[agent cfg]" in prompt
    assert "learning_rate=1.0e-3" in prompt


def test_discover_fields_parses_mock_llm_json() -> None:
    client = StubLLMClient(
        '{"project":{"task_id":"Template-Autotrain-v0","task_type":"ManagerBasedRLEnv"},'
        '"candidate_fields":[{"field":"algorithm.learning_rate","layer":"training","kind":"scalar",'
        '"source_file":"agent cfg","source_symbol":"cfg","default_value":0.001,'
        '"hydra_path":"algorithm.learning_rate","candidate_for_planner":true,"reason":"test"}],'
        '"absent_field_families":["command_ranges"]}'
    )

    payload = discover_fields(
        task_id="Template-Autotrain-v0",
        task_type="ManagerBasedRLEnv",
        source_snippets=[],
        llm_client=client,
    )

    assert payload["candidate_fields"][0]["field"] == "algorithm.learning_rate"


def test_discover_fields_extracts_json_from_fenced_response() -> None:
    client = StubLLMClient(
        "```json\n"
        '{"project":{"task_id":"Template-Autotrain-v0","task_type":"ManagerBasedRLEnv"},'
        '"candidate_fields":[],"absent_field_families":["command_ranges"]}'
        "\n```"
    )

    payload = discover_fields(
        task_id="Template-Autotrain-v0",
        task_type="ManagerBasedRLEnv",
        source_snippets=[],
        llm_client=client,
    )

    assert payload["absent_field_families"] == ["command_ranges"]


def test_discover_fields_rejects_unknown_absent_field_family() -> None:
    client = StubLLMClient(
        '{"project":{"task_id":"Template-Autotrain-v0","task_type":"ManagerBasedRLEnv"},'
        '"candidate_fields":[],"absent_field_families":["observation_noise"]}'
    )

    try:
        discover_fields(
            task_id="Template-Autotrain-v0",
            task_type="ManagerBasedRLEnv",
            source_snippets=[],
            llm_client=client,
        )
    except ValueError as exc:
        assert "Unsupported absent field family" in str(exc)
    else:
        raise AssertionError(
            "discover_fields should reject unsupported absent field families"
        )


def test_discover_fields_stabilizes_training_family_when_algorithm_fields_exist() -> (
    None
):
    client = StubLLMClient(
        '{"project":{"task_id":"Template-Autotrain-v0","task_type":"ManagerBasedRLEnv"},'
        '"candidate_fields":[{"field":"algorithm.learning_rate","layer":"training","kind":"scalar",'
        '"source_file":"agent cfg","source_symbol":"cfg","default_value":0.001,'
        '"hydra_path":"algorithm.learning_rate","candidate_for_planner":true,"reason":"test"}],'
        '"absent_field_families":["training","command_ranges"]}'
    )

    payload = discover_fields(
        task_id="Template-Autotrain-v0",
        task_type="ManagerBasedRLEnv",
        source_snippets=[],
        llm_client=client,
    )

    assert payload["absent_field_families"] == ["command_ranges"]


def test_load_source_snippet_reads_file_content(tmp_path: Path) -> None:
    source_path = tmp_path / "snippet.py"
    source_path.write_text("x = 1\n", encoding="utf-8")

    snippet = load_source_snippet(source_path, "env cfg")

    assert snippet["label"] == "env cfg"
    assert snippet["path"] == str(source_path)
    assert snippet["content"] == "x = 1\n"
