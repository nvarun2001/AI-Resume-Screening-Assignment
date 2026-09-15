from resume_screener.llm_adapter import LLMClient
from resume_screener.models import (
    EligibilityResult,
    GithubEnrichment,
    Project,
    ResumeExtraction,
)
from resume_screener.scoring import (
    assemble_score,
    score_ai_depth_deterministic,
    score_cloud_fullstack,
    score_engineering_depth,
    score_python_backend,
)


def _extraction(skills=None, projects=None, raw_text=""):
    return ResumeExtraction(
        source_file="c.pdf",
        skills=skills or [],
        projects=projects or [],
        raw_text=raw_text,
    )


def _eligible():
    return EligibilityResult(eligible=True, matched_skills=[])


# --- Category scoring ---

def test_python_backend_rewards_project_evidence_over_skills():
    skills_only = _extraction(skills=["Python", "FastAPI", "PostgreSQL", "Redis"])
    project_backed = _extraction(
        skills=["Python"],
        projects=[
            Project(
                name="API service",
                description="Async FastAPI service backed by PostgreSQL and Redis",
                technologies=["Python", "FastAPI", "PostgreSQL", "Redis"],
            )
        ],
    )
    skills_points, _ = score_python_backend(skills_only)
    project_points, _ = score_python_backend(project_backed)
    assert project_points > skills_points


def test_cloud_score_capped_at_max():
    extraction = _extraction(
        projects=[
            Project(
                name="Deploy",
                description="Dockerized app on GCP with Kubernetes, Terraform, CI/CD",
                technologies=["Docker", "GCP", "Kubernetes", "Terraform"],
            )
        ]
    )
    points, _ = score_cloud_fullstack(extraction)
    assert 0 < points <= 15


def test_engineering_depth_scores_signals():
    extraction = _extraction(
        projects=[
            Project(
                name="Platform",
                description="Added pytest suite, caching, a Kafka queue and Prometheus monitoring",
                technologies=["pytest", "kafka", "prometheus"],
            )
        ]
    )
    points, _ = score_engineering_depth(extraction)
    assert points > 0


def test_no_evidence_scores_zero():
    points, evidence = score_python_backend(_extraction(skills=["Photoshop"]))
    assert points == 0
    assert "No" in evidence


# --- AI project depth (deterministic) ---

def test_real_agentic_project_scores_higher_than_wrapper():
    rich = _extraction(
        projects=[
            Project(
                name="Support agent",
                description=(
                    "Multi-agent RAG workflow with retrieval, tool calling, state "
                    "orchestration and an evaluation pipeline"
                ),
                technologies=["LangGraph", "RAG", "embeddings"],
            )
        ]
    )
    wrapper = _extraction(
        projects=[
            Project(
                name="Chatbot",
                description="A simple chatbot using the OpenAI API",
                technologies=["OpenAI"],
            )
        ]
    )
    rich_score = score_ai_depth_deterministic(rich)
    wrapper_score = score_ai_depth_deterministic(wrapper)

    rich_final = rich_score.base_points - rich_score.penalty
    wrapper_final = wrapper_score.base_points - wrapper_score.penalty
    assert rich_final > wrapper_final
    assert rich_score.base_points >= 25


def test_wrapper_penalty_applied():
    wrapper = _extraction(
        projects=[
            Project(
                name="GPT clone",
                description="A ChatGPT clone that calls the OpenAI API",
                technologies=["OpenAI"],
            )
        ]
    )
    assessment = score_ai_depth_deterministic(wrapper)
    assert assessment.is_wrapper is True
    assert assessment.penalty > 0


def test_keyword_only_ai_scored_lower_than_project_backed():
    keyword_only = _extraction(skills=["LangChain", "RAG", "embeddings"])
    project_backed = _extraction(
        projects=[
            Project(
                name="RAG system",
                description="RAG pipeline with embeddings and retrieval",
                technologies=["LangChain"],
            )
        ]
    )
    assert (
        score_ai_depth_deterministic(keyword_only).base_points
        < score_ai_depth_deterministic(project_backed).base_points
    )


# --- assemble_score ---

def test_assemble_score_totals_and_github():
    extraction = _extraction(
        skills=["Python"],
        projects=[
            Project(
                name="Agent",
                description="RAG agent with retrieval and tool calling",
                technologies=["Python", "LangGraph", "FastAPI"],
            )
        ],
    )
    github = GithubEnrichment(status="ok", activity_points=4, repo_points=3, summary="Active")
    detail = assemble_score(extraction, _eligible(), github=github)
    assert detail.breakdown.github == 7
    assert 0 <= detail.breakdown.total <= 100
    assert detail.breakdown.ai_project_depth > 0
    assert detail.project_summary.startswith("Agent")


def test_assemble_score_uses_llm_when_client_given():
    def transport(system, user, schema):
        return {
            "base_points": 38,
            "penalty": 0,
            "is_wrapper": False,
            "evidence": "Strong agentic system",
            "strengths": ["Agentic workflow"],
            "concerns": [],
        }

    client = LLMClient(transport=transport)
    extraction = _extraction(
        skills=["Python"],
        projects=[Project(name="Agent", description="agentic system", technologies=["LangGraph"])],
    )
    detail = assemble_score(extraction, _eligible(), github=None, client=client)
    assert detail.breakdown.ai_project_depth == 38
    assert "Agentic workflow" in detail.strengths


def test_assemble_score_falls_back_when_llm_fails():
    def transport(system, user, schema):
        raise ValueError("model error")

    client = LLMClient(transport=transport)
    extraction = _extraction(
        projects=[Project(name="RAG", description="RAG retrieval pipeline", technologies=["LangChain"])]
    )
    detail = assemble_score(extraction, _eligible(), github=None, client=client)
    # Fell back to deterministic scoring, still produced a score.
    assert detail.breakdown.ai_project_depth > 0


def test_github_points_capped_at_10():
    extraction = _extraction(skills=["Python"])
    github = GithubEnrichment(status="ok", activity_points=5, repo_points=5, summary="x")
    detail = assemble_score(extraction, _eligible(), github=github)
    assert detail.breakdown.github == 10
