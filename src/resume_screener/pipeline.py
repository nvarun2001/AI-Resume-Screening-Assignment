"""End-to-end orchestration: ingest -> parse -> filter -> score -> rank.

Per-resume work (parse, extract, eligibility, score) runs sequentially and is
individually guarded so one bad resume never aborts the batch. GitHub lookups
for all eligible candidates run concurrently (bounded) since they are network
I/O bound. An LLM client is optional; without one the pipeline degrades to
heuristic extraction and deterministic scoring.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from .config import Settings
from .eligibility import evaluate_eligibility
from .extraction import extract_candidate
from .github_enrichment import GithubEnricher
from .ingestion import discover_resumes
from .llm_adapter import LLMClient
from .models import CandidateResult, ScreeningResult
from .parsing import parse_resume
from .ranking import build_summary, rank_candidates
from .scoring import assemble_score

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enricher: Optional[GithubEnricher] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or Settings.from_env()
        self.llm_client = llm_client
        self.enricher = enricher or GithubEnricher(self.settings)

    def run(self, input_dir: str | Path) -> ScreeningResult:
        files = discover_resumes(input_dir)
        logger.info("Screening %d resume(s)", len(files))

        results: list[CandidateResult] = []
        pending: list[tuple] = []  # (extraction, eligibility, name) for eligible candidates

        for path in files:
            parsed = parse_resume(path)
            fallback_name = Path(parsed.source_file).stem
            if parsed.status == "failed":
                results.append(
                    CandidateResult(
                        source_file=parsed.source_file,
                        candidate_name=fallback_name,
                        eligible=False,
                        parse_status="failed",
                        error=parsed.error,
                    )
                )
                continue

            extraction = extract_candidate(parsed, self.llm_client)
            eligibility = evaluate_eligibility(extraction)
            name = extraction.name or fallback_name

            if not eligibility.eligible:
                results.append(
                    CandidateResult(
                        source_file=extraction.source_file,
                        candidate_name=name,
                        eligible=False,
                        parse_status="ok",
                        rejection_reasons=eligibility.rejection_reasons,
                        matched_skills=eligibility.matched_skills,
                        email=extraction.email,
                        github_username=extraction.github_username,
                    )
                )
                continue

            pending.append((extraction, eligibility, name))

        github_map = self._enrich_github([ctx[0].github_username for ctx in pending])

        for extraction, eligibility, name in pending:
            github = github_map.get(extraction.github_username) if extraction.github_username else None
            detail = assemble_score(extraction, eligibility, github=github, client=self.llm_client)
            results.append(
                CandidateResult(
                    source_file=extraction.source_file,
                    candidate_name=name,
                    eligible=True,
                    parse_status="ok",
                    total_score=detail.breakdown.total,
                    score_breakdown=detail.breakdown,
                    project_summary=detail.project_summary,
                    strengths=detail.strengths,
                    concerns=detail.concerns,
                    matched_skills=eligibility.matched_skills,
                    email=extraction.email,
                    github_username=extraction.github_username,
                    github_summary=github.summary if github else "No GitHub profile provided",
                )
            )

        ranked = rank_candidates(results)
        return ScreeningResult(summary=build_summary(ranked), candidates=ranked)

    def _enrich_github(self, usernames: list[Optional[str]]):
        if not any(usernames):
            return {}
        try:
            return asyncio.run(self.enricher.enrich_many(usernames))
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
            logger.warning("GitHub enrichment batch failed: %s", exc)
            return {}
