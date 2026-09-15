"""Pydantic models shared across the pipeline.

These types are the contract between stages: parsing/extraction produce a
``ResumeExtraction``, eligibility and scoring consume it, and the pipeline
assembles everything into a ``CandidateResult`` that serialises directly to
the output JSON.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Project(BaseModel):
    """A single project pulled from a resume."""

    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class WorkExperience(BaseModel):
    """An internship or job entry."""

    organization: str = ""
    role: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class ResumeFields(BaseModel):
    """Fields the LLM is asked to extract from raw resume text.

    Kept separate from ``ResumeExtraction`` so the LLM schema stays small and
    focused. Email/GitHub are intentionally omitted here because regex handles
    them more reliably than a language model.
    """

    name: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)


class ResumeExtraction(BaseModel):
    """The merged, structured view of one resume.

    Combines regex-derived atomic fields (email, GitHub) with the LLM-extracted
    skills/projects/experience. ``extraction_status`` records whether the LLM
    step succeeded so downstream stages can account for degraded data.
    """

    source_file: str
    name: Optional[str] = None
    email: Optional[str] = None
    github_url: Optional[str] = None
    github_username: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    raw_text: str = ""
    extraction_status: str = "ok"  # ok | llm_failed


class AIDepthAssessment(BaseModel):
    """Structured judgment of AI/agentic project depth from the LLM.

    ``base_points`` is the merit before penalties; ``penalty`` captures the
    deduction for shallow wrapper/tutorial work. Keeping them separate makes the
    final score auditable.
    """

    base_points: int = 0
    penalty: int = 0
    is_wrapper: bool = False
    evidence: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class EligibilityResult(BaseModel):
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    python_evidence: Optional[str] = None
    ai_evidence: Optional[str] = None


class GithubEnrichment(BaseModel):
    """Result of the public GitHub lookup for one candidate."""

    status: str  # ok | not_provided | not_found | rate_limited | error
    username: Optional[str] = None
    activity_points: int = 0
    repo_points: int = 0
    summary: str = ""

    @property
    def total_points(self) -> int:
        return self.activity_points + self.repo_points


class ScoreBreakdown(BaseModel):
    """The five scoring categories. Serialises to match the required schema."""

    ai_project_depth: int = 0
    python_backend: int = 0
    cloud_fullstack: int = 0
    github: int = 0
    engineering_depth: int = 0

    @property
    def total(self) -> int:
        return (
            self.ai_project_depth
            + self.python_backend
            + self.cloud_fullstack
            + self.github
            + self.engineering_depth
        )


class ScoreDetail(BaseModel):
    """Everything the scorer produces for an eligible candidate.

    The numeric breakdown is separated from the narrative (evidence, strengths,
    concerns) so the output stays explainable without bloating the score object.
    """

    breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    project_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class CandidateResult(BaseModel):
    """Final per-candidate record written to the output JSON."""

    source_file: str
    candidate_name: str
    eligible: bool
    parse_status: str = "ok"  # ok | failed

    # Populated for eligible candidates.
    rank: Optional[int] = None
    total_score: Optional[int] = None
    score_breakdown: Optional[ScoreBreakdown] = None
    project_summary: Optional[str] = None
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    # Populated for rejected candidates.
    rejection_reasons: list[str] = Field(default_factory=list)

    # Common to both outcomes when available.
    matched_skills: list[str] = Field(default_factory=list)
    email: Optional[str] = None
    github_username: Optional[str] = None
    github_summary: Optional[str] = None
    error: Optional[str] = None


class BatchSummary(BaseModel):
    total_resumes: int = 0
    successfully_parsed: int = 0
    eligible: int = 0
    rejected: int = 0
    failed: int = 0


class ScreeningResult(BaseModel):
    """Top-level output: the ranked list plus the batch summary."""

    summary: BatchSummary
    candidates: list[CandidateResult] = Field(default_factory=list)
