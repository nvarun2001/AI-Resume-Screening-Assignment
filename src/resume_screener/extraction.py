"""LLM-driven structured field extraction from raw resume text.

The LLM turns messy resume text into a ``ResumeFields`` object (skills,
projects, work experience). Atomic fields (email, GitHub) come from regex and
always win over the model. If the model call fails, a keyword-based heuristic
fills in enough for the deterministic eligibility check to still run, and the
result is flagged ``llm_failed`` so downstream stages know the data is partial.
"""

from __future__ import annotations

import logging
from typing import Optional

from . import config
from .llm_adapter import LLMClient, LLMError
from .models import ResumeExtraction, ResumeFields
from .parsing import ParsedResume

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured information from a software engineering resume. "
    "Only record information explicitly present in the text. Do not infer or "
    "invent skills, projects, or experience. For each project, capture the "
    "technologies actually mentioned and a short factual description. Leave a "
    "field empty if the resume does not state it."
)


def extract_candidate(parsed: ParsedResume, client: Optional[LLMClient] = None) -> ResumeExtraction:
    """Produce a merged ``ResumeExtraction`` from parsed text + the LLM.

    When ``client`` is ``None`` the LLM step is skipped entirely and a
    keyword-based heuristic is used, so the pipeline can run without an LLM.
    """

    text = parsed.text[: config.MAX_RESUME_CHARS]
    if client is None:
        fields = _heuristic_fields(parsed.text)
        status = "no_llm"
    else:
        try:
            fields = client.extract_structured(
                EXTRACTION_SYSTEM_PROMPT,
                f"Resume text:\n\n{text}",
                ResumeFields,
            )
            status = "ok"
        except LLMError as exc:
            logger.warning("LLM extraction failed for %s: %s", parsed.source_file, exc)
            fields = _heuristic_fields(parsed.text)
            status = "llm_failed"

    name = fields.name or _guess_name(parsed.text)
    return ResumeExtraction(
        source_file=parsed.source_file,
        name=name,
        email=parsed.email,
        github_url=parsed.github_url,
        github_username=parsed.github_username,
        skills=fields.skills,
        projects=fields.projects,
        work_experience=fields.work_experience,
        raw_text=parsed.text,
        extraction_status=status,
    )


def _heuristic_fields(text: str) -> ResumeFields:
    """Best-effort extraction without the LLM.

    Scans the text for known technology keywords so eligibility/scoring have
    something to work with. Projects/experience can't be reconstructed reliably
    this way, so they stay empty and the caller records the degraded status.
    """

    lowered = text.lower()
    vocab = (
        config.PYTHON_KEYWORDS
        + config.AI_KEYWORDS
        + config.BACKEND_KEYWORDS
        + config.CLOUD_KEYWORDS
        + config.ENGINEERING_DEPTH_KEYWORDS
    )
    seen: list[str] = []
    for keyword in vocab:
        if keyword in lowered and keyword not in seen:
            seen.append(keyword)
    return ResumeFields(name=_guess_name(text), skills=seen, projects=[], work_experience=[])


def _guess_name(text: str) -> str | None:
    """Assume the candidate's name is the first plausible line of the resume."""

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "@" in stripped or "http" in stripped.lower():
            continue
        if len(stripped) > 60:
            continue
        if any(ch.isdigit() for ch in stripped):
            continue
        return stripped
    return None
