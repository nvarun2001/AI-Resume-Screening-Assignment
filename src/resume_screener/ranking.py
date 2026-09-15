"""Rank eligible candidates and build the batch summary."""

from __future__ import annotations

from .models import BatchSummary, CandidateResult


def rank_candidates(results: list[CandidateResult]) -> list[CandidateResult]:
    """Order results: ranked eligible (highest score first), then rejected, then failed.

    Ranks are assigned only to eligible candidates. Ties break alphabetically by
    name for stable, reproducible output.
    """

    eligible = [r for r in results if r.eligible]
    rejected = [r for r in results if not r.eligible and r.parse_status != "failed"]
    failed = [r for r in results if r.parse_status == "failed"]

    eligible.sort(key=lambda r: (-(r.total_score or 0), r.candidate_name.lower()))
    for position, result in enumerate(eligible, start=1):
        result.rank = position

    return eligible + rejected + failed


def build_summary(results: list[CandidateResult]) -> BatchSummary:
    total = len(results)
    failed = sum(1 for r in results if r.parse_status == "failed")
    eligible = sum(1 for r in results if r.eligible)
    rejected = sum(1 for r in results if not r.eligible and r.parse_status != "failed")
    return BatchSummary(
        total_resumes=total,
        successfully_parsed=total - failed,
        eligible=eligible,
        rejected=rejected,
        failed=failed,
    )
