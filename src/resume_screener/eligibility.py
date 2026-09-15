"""Deterministic hard eligibility filter (Python + AI/agentic evidence).

This gate is intentionally rule-based and LLM-free so it is predictable and
testable. It operates on the already-extracted structured fields plus the raw
text. A candidate is eligible only when there is evidence of BOTH Python and a
meaningful AI/LLM/agentic technology. Depth is judged later by scoring; here we
only decide relevance.
"""

from __future__ import annotations

from . import config
from .matching import contains
from .models import EligibilityResult, ResumeExtraction

# Full vocabulary of technologies we recognise, used to filter the candidate's
# declared skills down to meaningful matches for the output.
_KNOWN_TECH = {
    term.lower()
    for term in (
        config.PYTHON_EVIDENCE_TERMS
        + config.AI_KEYWORDS
        + config.BACKEND_KEYWORDS
        + config.CLOUD_KEYWORDS
        + config.ENGINEERING_DEPTH_KEYWORDS
    )
}


def _first_match(corpus: str, terms: list[str]) -> str | None:
    for term in terms:
        if contains(corpus, term):
            return term
    return None


def _build_corpus(extraction: ResumeExtraction) -> str:
    """Combine every field that could carry technology evidence."""

    parts: list[str] = list(extraction.skills)
    for project in extraction.projects:
        parts.extend([project.name, project.description])
        parts.extend(project.technologies)
    for exp in extraction.work_experience:
        parts.extend([exp.role, exp.description])
        parts.extend(exp.technologies)
    parts.append(extraction.raw_text)
    return "\n".join(p for p in parts if p).lower()


def evaluate_eligibility(extraction: ResumeExtraction) -> EligibilityResult:
    corpus = _build_corpus(extraction)

    python_hit = _first_match(corpus, config.PYTHON_EVIDENCE_TERMS)
    ai_hit = _first_match(corpus, config.AI_KEYWORDS)

    reasons: list[str] = []
    if not python_hit:
        reasons.append(
            "No evidence of Python as a skill, project technology, or implementation language"
        )
    if not ai_hit:
        reasons.append("No AI/LLM/RAG/agentic project or framework evidence")

    matched_skills = [
        skill for skill in extraction.skills if skill.lower() in _KNOWN_TECH
    ]

    return EligibilityResult(
        eligible=not reasons,
        rejection_reasons=reasons,
        matched_skills=matched_skills,
        python_evidence=python_hit,
        ai_evidence=ai_hit,
    )
