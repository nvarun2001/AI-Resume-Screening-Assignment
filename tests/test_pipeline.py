from resume_screener.github_enrichment import GithubEnricher
from resume_screener.pipeline import Pipeline


ELIGIBLE_RESUME = """Asha Rao
Software Engineer

Projects:
Built a multi-agent RAG pipeline in Python with LangGraph, retrieval,
tool calling and an evaluation harness. Backend in FastAPI with PostgreSQL
and Redis. Deployed with Docker on GCP. Added pytest suite and monitoring.

Contact: asha@example.com  github.com/asharao
"""

REJECTED_RESUME = """Bob Jones
Frontend Developer

Skills: Java, React, Spring Boot, JavaScript
Built web applications and REST endpoints.
bob@example.com
"""


def _make_resumes(tmp_path):
    (tmp_path / "candidate_01.txt").write_text(ELIGIBLE_RESUME, encoding="utf-8")
    (tmp_path / "candidate_02.txt").write_text(REJECTED_RESUME, encoding="utf-8")
    (tmp_path / "candidate_03.txt").write_text("   ", encoding="utf-8")  # unreadable -> failed


def _enricher():
    async def transport(path):
        if "events" in path:
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc).isoformat()
            return 200, [{"created_at": now} for _ in range(8)]
        return 200, [{"fork": False, "language": "Python", "name": "rag-agent"}]

    return GithubEnricher(transport=transport)


def test_pipeline_end_to_end(tmp_path):
    _make_resumes(tmp_path)
    pipeline = Pipeline(llm_client=None, enricher=_enricher())
    result = pipeline.run(tmp_path)

    assert result.summary.total_resumes == 3
    assert result.summary.eligible == 1
    assert result.summary.rejected == 1
    assert result.summary.failed == 1
    assert result.summary.successfully_parsed == 2

    top = result.candidates[0]
    assert top.candidate_name == "Asha Rao"
    assert top.eligible is True
    assert top.rank == 1
    assert top.total_score and top.total_score > 0
    assert top.score_breakdown.github > 0  # enrichment folded in
    assert "events" in (top.github_summary or "")


def test_pipeline_rejected_has_reasons(tmp_path):
    _make_resumes(tmp_path)
    pipeline = Pipeline(llm_client=None, enricher=_enricher())
    result = pipeline.run(tmp_path)

    rejected = [c for c in result.candidates if not c.eligible and c.parse_status == "ok"]
    assert len(rejected) == 1
    assert rejected[0].rejection_reasons


def test_pipeline_empty_directory(tmp_path):
    result = Pipeline(llm_client=None, enricher=_enricher()).run(tmp_path)
    assert result.summary.total_resumes == 0
    assert result.candidates == []
