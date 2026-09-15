from datetime import datetime, timedelta, timezone

import pytest

from resume_screener.config import Settings
from resume_screener.github_enrichment import (
    GithubEnricher,
    score_activity,
    score_repos,
)


def _events(n, days_ago=1):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return [{"created_at": ts} for _ in range(n)]


# --- pure scoring functions ---

def test_score_activity_recent_events():
    assert score_activity(_events(10)) == 5
    assert score_activity(_events(4)) == 3
    assert score_activity(_events(0)) == 0


def test_score_activity_ignores_old_events():
    old = _events(10, days_ago=400)
    assert score_activity(old) == 0


def test_score_repos_counts_non_forks():
    repos = [
        {"fork": False, "language": "Python", "name": "a"},
        {"fork": False, "language": "Go", "name": "b"},
        {"fork": True, "language": "Python", "name": "c"},
    ]
    points, relevant = score_repos(repos)
    assert relevant is True  # a Python repo present
    assert points > 0


def test_score_repos_relevance_from_ai_keyword():
    repos = [{"fork": False, "language": "TypeScript", "name": "rag-agent", "description": "LangChain bot"}]
    _, relevant = score_repos(repos)
    assert relevant is True


def test_score_repos_no_relevant():
    repos = [{"fork": False, "language": "Java", "name": "banking-app", "description": "spring"}]
    points, relevant = score_repos(repos)
    assert relevant is False


# --- async enrichment with injected transport ---

@pytest.mark.asyncio
async def test_enrich_not_provided():
    enricher = GithubEnricher(transport=_never_called)
    result = await enricher.enrich(None)
    assert result.status == "not_provided"
    assert result.total_points == 0


@pytest.mark.asyncio
async def test_enrich_success():
    async def transport(path):
        if "events" in path:
            return 200, _events(10)
        return 200, [{"fork": False, "language": "Python", "name": "ml-agent"}]

    enricher = GithubEnricher(transport=transport)
    result = await enricher.enrich("janedoe")
    assert result.status == "ok"
    assert result.activity_points == 5
    assert result.repo_points > 0


@pytest.mark.asyncio
async def test_enrich_404_not_found():
    async def transport(path):
        return 404, {"message": "Not Found"}

    result = await GithubEnricher(transport=transport).enrich("ghost")
    assert result.status == "not_found"
    assert result.total_points == 0


@pytest.mark.asyncio
async def test_enrich_rate_limited():
    async def transport(path):
        return 403, {"message": "rate limit exceeded"}

    result = await GithubEnricher(transport=transport).enrich("someone")
    assert result.status == "rate_limited"
    assert result.total_points == 0


@pytest.mark.asyncio
async def test_enrich_handles_exception():
    async def transport(path):
        raise RuntimeError("network down")

    result = await GithubEnricher(transport=transport).enrich("someone")
    assert result.status == "error"
    assert result.total_points == 0


@pytest.mark.asyncio
async def test_enrich_caches_per_username():
    calls = {"n": 0}

    async def transport(path):
        calls["n"] += 1
        if "events" in path:
            return 200, _events(3)
        return 200, []

    enricher = GithubEnricher(transport=transport)
    await enricher.enrich("dup")
    await enricher.enrich("dup")
    # 2 endpoints for the first call only; the second is served from cache.
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_enrich_many_dedups():
    async def transport(path):
        if "events" in path:
            return 200, _events(1)
        return 200, []

    enricher = GithubEnricher(transport=transport)
    results = await enricher.enrich_many(["a", "a", "b", None])
    assert set(results) == {"a", "b"}


async def _never_called(path):
    raise AssertionError("transport should not be called")
