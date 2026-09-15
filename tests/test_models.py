import os

from resume_screener.config import (
    AI_PROJECT_DEPTH_MAX,
    CLOUD_FULLSTACK_MAX,
    ENGINEERING_DEPTH_MAX,
    GITHUB_MAX,
    PYTHON_BACKEND_MAX,
    Settings,
)
from resume_screener.models import (
    GithubEnrichment,
    Project,
    ResumeExtraction,
    ScoreBreakdown,
)


def test_weights_sum_to_100():
    total = (
        AI_PROJECT_DEPTH_MAX
        + PYTHON_BACKEND_MAX
        + CLOUD_FULLSTACK_MAX
        + GITHUB_MAX
        + ENGINEERING_DEPTH_MAX
    )
    assert total == 100


def test_resume_extraction_defaults():
    extraction = ResumeExtraction(source_file="candidate_01.pdf")
    assert extraction.skills == []
    assert extraction.projects == []
    assert extraction.extraction_status == "ok"
    assert extraction.email is None


def test_project_technologies_list():
    project = Project(name="RAG bot", technologies=["Python", "LangChain"])
    assert "LangChain" in project.technologies


def test_score_breakdown_total():
    breakdown = ScoreBreakdown(
        ai_project_depth=35,
        python_backend=27,
        cloud_fullstack=12,
        github=8,
        engineering_depth=4,
    )
    assert breakdown.total == 86


def test_github_enrichment_total_points():
    enrichment = GithubEnrichment(status="ok", activity_points=4, repo_points=3)
    assert enrichment.total_points == 7


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "8")
    settings = Settings.from_env()
    assert settings.llm_provider == "anthropic"
    assert settings.llm_configured is True
    assert settings.llm_max_concurrency == 8


def test_settings_not_configured_when_key_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = Settings.from_env()
    assert settings.llm_configured is False


def test_settings_bad_concurrency_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "not-a-number")
    settings = Settings.from_env()
    assert settings.llm_max_concurrency == 5
