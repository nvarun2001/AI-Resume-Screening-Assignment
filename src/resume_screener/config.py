"""Central configuration: scoring weights, keyword lists, env-driven settings.

Everything tunable lives here so business logic never hard-codes a weight,
threshold, or keyword. Runtime secrets and provider choice come from the
environment (loaded from a local ``.env`` when present).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# Upper bound on resume text sent to the model, to keep token usage predictable.
MAX_RESUME_CHARS = 15000


# --- LLM request tuning ---
# Temperature 0 keeps extraction/scoring as reproducible as the model allows,
# which matters for a screening tool that must be explainable and re-runnable.
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.0


# --- Scoring weights (must sum to 100) ---
AI_PROJECT_DEPTH_MAX = 40
PYTHON_BACKEND_MAX = 30
CLOUD_FULLSTACK_MAX = 15
GITHUB_MAX = 10
ENGINEERING_DEPTH_MAX = 5


# --- Keyword vocabularies ---
# Lower-cased; matching is done case-insensitively against skills, project
# technologies, and free text.

PYTHON_KEYWORDS = [
    "python",
    "py",
    "django",
    "flask",
    "fastapi",
    "pandas",
    "numpy",
    "pytorch",
    "tensorflow",
]

# A hit on any of these signals a genuine AI/agentic/LLM project.
AI_KEYWORDS = [
    "llm",
    "large language model",
    "langchain",
    "langgraph",
    "llamaindex",
    "llama index",
    "google adk",
    "agentic",
    "agent",
    "multi-agent",
    "rag",
    "retrieval augmented",
    "retrieval-augmented",
    "embedding",
    "embeddings",
    "vector search",
    "vector database",
    "vector db",
    "semantic search",
    "tool calling",
    "tool-calling",
    "function calling",
    "prompt engineering",
    "fine-tuning",
    "fine tuning",
    "openai",
    "anthropic",
    "claude",
    "gpt",
    "hugging face",
    "huggingface",
    "transformers",
    "gemini",
    "bedrock",
]

# Terms that count as genuine Python evidence for the eligibility gate. A
# Python-specific framework (Django/Flask/FastAPI/...) implies Python even if
# the word "python" is absent. Deliberately excludes the loose "py" token.
PYTHON_EVIDENCE_TERMS = [
    "python",
    "django",
    "flask",
    "fastapi",
    "pytorch",
    "tensorflow",
    "pandas",
    "numpy",
]

BACKEND_KEYWORDS = [
    "fastapi",
    "django",
    "flask",
    "asyncio",
    "async",
    "await",
    "postgresql",
    "postgres",
    "psycopg",
    "redis",
    "celery",
    "sqlalchemy",
    "rest api",
    "restful",
    "graphql",
    "grpc",
    "microservice",
    "microservices",
]

CLOUD_KEYWORDS = [
    "gcp",
    "google cloud",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "k8s",
    "terraform",
    "ci/cd",
    "cicd",
    "github actions",
    "cloud run",
    "lambda",
    "s3",
    "ec2",
    "react",
    "next.js",
    "nextjs",
    "node.js",
    "nodejs",
    "vercel",
]

ENGINEERING_DEPTH_KEYWORDS = [
    "test",
    "testing",
    "pytest",
    "unittest",
    "unit test",
    "integration test",
    "caching",
    "cache",
    "queue",
    "kafka",
    "rabbitmq",
    "sqs",
    "observability",
    "monitoring",
    "prometheus",
    "grafana",
    "logging",
    "concurrency",
    "load balancing",
    "rate limiting",
    "retry",
    "circuit breaker",
    "profiling",
]

# Words that hint a "project" is really a shallow API wrapper. Used to justify
# the project-quality penalty when no depth signals are present.
WRAPPER_HINT_KEYWORDS = [
    "chatbot using",
    "chatgpt clone",
    "wrapper",
    "simple chatbot",
    "api call",
    "calls the openai api",
    "using openai api",
]

# Signals that a project has real depth (retrieval, state, orchestration...).
DEPTH_SIGNAL_KEYWORDS = [
    "retrieval",
    "rag",
    "vector",
    "embedding",
    "state",
    "orchestration",
    "tool",
    "evaluation",
    "eval",
    "pipeline",
    "multi-agent",
    "agentic",
    "workflow",
    "memory",
    "fine-tun",
]


# --- GitHub enrichment scoring ---
GITHUB_API_BASE = "https://api.github.com"
GITHUB_ACTIVITY_LOOKBACK_DAYS = 90
GITHUB_REQUEST_TIMEOUT = 10.0

# Recent public events in the lookback window -> activity points (0-5).
# Each tuple is (minimum_event_count, points); evaluated high-to-low.
GITHUB_ACTIVITY_THRESHOLDS = [
    (10, 5),
    (5, 4),
    (3, 3),
    (1, 2),
    (0, 0),
]

# Non-fork, non-empty owned repos -> base repo points (0-3), plus up to 2 bonus
# points when any repo is Python/AI relevant.
GITHUB_REPO_COUNT_THRESHOLDS = [
    (5, 3),
    (2, 2),
    (1, 1),
    (0, 0),
]
GITHUB_RELEVANT_REPO_BONUS = 2


@dataclass
class Settings:
    """Runtime configuration sourced from environment variables."""

    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    github_token: str = ""
    llm_max_concurrency: int = 5
    github_max_concurrency: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
            github_token=os.getenv("GITHUB_TOKEN", "").strip(),
            llm_max_concurrency=_int_env("LLM_MAX_CONCURRENCY", 5),
            github_max_concurrency=_int_env("GITHUB_MAX_CONCURRENCY", 5),
        )

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_provider and self.llm_model and self.llm_api_key)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default
