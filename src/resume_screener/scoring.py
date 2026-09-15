"""100-point scoring rubric with project-quality penalties.

Hybrid design:

- Python/backend, cloud/full-stack and engineering-depth are scored
  deterministically from keyword evidence, weighting matches that appear in a
  project or work description over skills-list-only mentions.
- AI project depth (the largest category) is judged by the LLM when available,
  because "real agentic system vs. thin API wrapper" needs semantic judgment.
  A deterministic fallback keeps the pipeline working without an LLM.
- GitHub points come from the enrichment step and are simply folded in here.

Every category returns an evidence string so the final score is explainable.
"""

from __future__ import annotations

import logging
from typing import Optional

from . import config
from .llm_adapter import LLMClient, LLMError
from .matching import contains, distinct_matches
from .models import (
    AIDepthAssessment,
    EligibilityResult,
    GithubEnrichment,
    ResumeExtraction,
    ScoreBreakdown,
    ScoreDetail,
)

logger = logging.getLogger(__name__)

AI_DEPTH_SYSTEM_PROMPT = (
    "You assess the DEPTH of a candidate's AI/agentic/LLM work for a 40-point "
    "category. Award base_points (0-40) for genuine depth: agents, RAG, "
    "retrieval, tool calling, state/orchestration, evaluation, and real product "
    "or business logic. Set is_wrapper true and a penalty of 5-15 when a "
    "project is only a thin wrapper around an LLM API call, or a tutorial-style "
    "clone with no evidence of ownership or engineering. Base your judgment only "
    "on what the text states, and give concrete evidence, strengths, and concerns."
)


def _project_work_corpus(extraction: ResumeExtraction) -> str:
    parts: list[str] = []
    for project in extraction.projects:
        parts.extend([project.name, project.description])
        parts.extend(project.technologies)
    for exp in extraction.work_experience:
        parts.extend([exp.role, exp.description])
        parts.extend(exp.technologies)
    corpus = "\n".join(p for p in parts if p).lower()
    # Without structured projects (e.g. deterministic mode or a failed LLM
    # extraction) fall back to the raw text so evidence isn't lost and the
    # candidate isn't wrongly treated as "skills-list only".
    if not corpus:
        return extraction.raw_text.lower()
    return corpus


def _full_corpus(extraction: ResumeExtraction) -> str:
    parts = list(extraction.skills)
    parts.append(_project_work_corpus(extraction))
    parts.append(extraction.raw_text)
    return "\n".join(p for p in parts if p).lower()


def _category_score(
    extraction: ResumeExtraction, keywords: list[str], max_points: int, name: str
) -> tuple[int, str]:
    full = _full_corpus(extraction)
    project_work = _project_work_corpus(extraction)

    matches = distinct_matches(full, keywords)
    if not matches:
        return 0, f"No {name.replace('_', '/')} evidence found"

    project_backed = any(contains(project_work, term) for term in matches)
    factor = 1.0 if project_backed else config.KEYWORD_ONLY_FACTOR
    saturation = config.CATEGORY_SATURATION[name]
    points = round(max_points * min(1.0, len(matches) / saturation) * factor)
    points = min(points, max_points)

    where = "project/experience-backed" if project_backed else "skills-list only"
    return points, f"Matched {matches} ({where})"


def score_python_backend(extraction: ResumeExtraction) -> tuple[int, str]:
    keywords = config.PYTHON_EVIDENCE_TERMS + config.BACKEND_KEYWORDS
    return _category_score(extraction, keywords, config.PYTHON_BACKEND_MAX, "python_backend")


def score_cloud_fullstack(extraction: ResumeExtraction) -> tuple[int, str]:
    return _category_score(
        extraction, config.CLOUD_KEYWORDS, config.CLOUD_FULLSTACK_MAX, "cloud_fullstack"
    )


def score_engineering_depth(extraction: ResumeExtraction) -> tuple[int, str]:
    return _category_score(
        extraction, config.ENGINEERING_DEPTH_KEYWORDS, config.ENGINEERING_DEPTH_MAX, "engineering_depth"
    )


def score_ai_depth_deterministic(extraction: ResumeExtraction) -> AIDepthAssessment:
    full = _full_corpus(extraction)
    project_work = _project_work_corpus(extraction)

    ai_matches = distinct_matches(full, config.AI_KEYWORDS)
    depth_matches = distinct_matches(full, config.DEPTH_SIGNAL_KEYWORDS)

    weighted = (
        len(depth_matches) * config.AI_DEPTH_SIGNAL_WEIGHT
        + len(ai_matches) * config.AI_KEYWORD_WEIGHT
    )
    base = round(config.AI_PROJECT_DEPTH_MAX * min(1.0, weighted / config.AI_DEPTH_SATURATION))

    project_backed = any(contains(project_work, term) for term in ai_matches + depth_matches)
    if not project_backed:
        base = round(base * config.KEYWORD_ONLY_FACTOR)

    wrapper_hit = any(contains(full, hint) for hint in config.WRAPPER_HINT_KEYWORDS)
    is_wrapper = wrapper_hit and not depth_matches
    penalty = config.AI_WRAPPER_PENALTY if is_wrapper else 0

    strengths: list[str] = []
    concerns: list[str] = []
    if depth_matches:
        strengths.append("Depth signals: " + ", ".join(depth_matches))
    if ai_matches:
        strengths.append("AI stack: " + ", ".join(ai_matches))
    if is_wrapper:
        concerns.append("Appears to be a thin LLM API wrapper with limited engineering depth")
    if not project_backed and ai_matches:
        concerns.append("AI keywords appear without clear project evidence")

    evidence = (
        f"ai_keywords={ai_matches}; depth_signals={depth_matches}; "
        f"project_backed={project_backed}"
    )
    return AIDepthAssessment(
        base_points=base,
        penalty=penalty,
        is_wrapper=is_wrapper,
        evidence=evidence,
        strengths=strengths,
        concerns=concerns,
    )


def _ai_depth_via_llm(extraction: ResumeExtraction, client: LLMClient) -> AIDepthAssessment:
    project_lines = [
        f"- {p.name}: {p.description} (tech: {', '.join(p.technologies)})"
        for p in extraction.projects
    ]
    user = "Skills: {skills}\n\nProjects:\n{projects}".format(
        skills=", ".join(extraction.skills) or "(none listed)",
        projects="\n".join(project_lines) or "(no structured projects)",
    )
    return client.extract_structured(AI_DEPTH_SYSTEM_PROMPT, user, AIDepthAssessment)


def assemble_score(
    extraction: ResumeExtraction,
    eligibility: EligibilityResult,
    github: Optional[GithubEnrichment] = None,
    client: Optional[LLMClient] = None,
) -> ScoreDetail:
    """Combine all categories into a ``ScoreDetail`` for one eligible candidate."""

    assessment: Optional[AIDepthAssessment] = None
    if client is not None:
        try:
            assessment = _ai_depth_via_llm(extraction, client)
        except LLMError as exc:
            logger.warning("AI-depth LLM scoring failed for %s: %s", extraction.source_file, exc)
    if assessment is None:
        assessment = score_ai_depth_deterministic(extraction)

    base = max(0, min(config.AI_PROJECT_DEPTH_MAX, assessment.base_points))
    penalty = max(0, min(config.AI_WRAPPER_PENALTY + 5, assessment.penalty))
    ai_points = max(0, base - penalty)

    python_backend, pb_ev = score_python_backend(extraction)
    cloud_fullstack, cf_ev = score_cloud_fullstack(extraction)
    engineering_depth, ed_ev = score_engineering_depth(extraction)
    github_points = min(config.GITHUB_MAX, github.total_points) if github else 0

    breakdown = ScoreBreakdown(
        ai_project_depth=ai_points,
        python_backend=python_backend,
        cloud_fullstack=cloud_fullstack,
        github=github_points,
        engineering_depth=engineering_depth,
    )

    evidence = {
        "ai_project_depth": f"base={base}, penalty={penalty}. {assessment.evidence}",
        "python_backend": pb_ev,
        "cloud_fullstack": cf_ev,
        "engineering_depth": ed_ev,
        "github": github.summary if github else "No GitHub profile provided",
    }

    return ScoreDetail(
        breakdown=breakdown,
        project_summary=_project_summary(extraction),
        strengths=assessment.strengths,
        concerns=assessment.concerns,
        evidence=evidence,
    )


def _project_summary(extraction: ResumeExtraction) -> str:
    if extraction.projects:
        project = extraction.projects[0]
        detail = project.description or ", ".join(project.technologies)
        return f"{project.name}: {detail}".strip().strip(":").strip()
    # Fallback for deterministic mode: a short snippet of the raw text.
    collapsed = " ".join(extraction.raw_text.split())
    return collapsed[:200]
