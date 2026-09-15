from resume_screener.models import CandidateResult, ScoreBreakdown
from resume_screener.ranking import build_summary, rank_candidates


def _eligible(name, score):
    return CandidateResult(
        source_file=f"{name}.pdf",
        candidate_name=name,
        eligible=True,
        parse_status="ok",
        total_score=score,
        score_breakdown=ScoreBreakdown(ai_project_depth=score),
    )


def _rejected(name):
    return CandidateResult(
        source_file=f"{name}.pdf",
        candidate_name=name,
        eligible=False,
        parse_status="ok",
        rejection_reasons=["No Python"],
    )


def _failed(name):
    return CandidateResult(
        source_file=f"{name}.pdf",
        candidate_name=name,
        eligible=False,
        parse_status="failed",
        error="unreadable",
    )


def test_eligible_sorted_desc_and_ranked():
    results = [_eligible("A", 70), _eligible("B", 90), _eligible("C", 80)]
    ranked = rank_candidates(results)
    assert [r.candidate_name for r in ranked] == ["B", "C", "A"]
    assert [r.rank for r in ranked] == [1, 2, 3]


def test_tie_breaks_alphabetically():
    results = [_eligible("Zoe", 80), _eligible("Ana", 80)]
    ranked = rank_candidates(results)
    assert [r.candidate_name for r in ranked] == ["Ana", "Zoe"]


def test_rejected_and_failed_after_eligible_without_rank():
    results = [_rejected("R"), _eligible("E", 50), _failed("F")]
    ranked = rank_candidates(results)
    assert ranked[0].candidate_name == "E"
    assert ranked[0].rank == 1
    non_eligible = [r for r in ranked if not r.eligible]
    assert all(r.rank is None for r in non_eligible)


def test_build_summary_counts():
    results = [_eligible("E", 50), _rejected("R"), _failed("F"), _eligible("E2", 60)]
    summary = build_summary(results)
    assert summary.total_resumes == 4
    assert summary.eligible == 2
    assert summary.rejected == 1
    assert summary.failed == 1
    assert summary.successfully_parsed == 3
