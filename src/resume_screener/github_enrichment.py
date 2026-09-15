"""Lightweight public GitHub activity signal (0-10 points).

Two public endpoints per candidate: recent public events (activity, 0-5) and
owned repositories (maintained/relevant, 0-5). GitHub is an additive signal
only, never an eligibility gate, so every failure mode (missing profile, 404,
rate limit, network error) degrades to a recorded status with zero points and
never raises. Results are cached per run and lookups are concurrency-bounded.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

from . import config
from .matching import contains
from .models import GithubEnrichment
from .config import Settings

logger = logging.getLogger(__name__)

# transport(path) -> (status_code, parsed_json); injected in tests.
Transport = Callable[[str], Awaitable[tuple[int, object]]]


def score_activity(events: list[dict], now: Optional[datetime] = None) -> int:
    """Points (0-5) from public events within the lookback window."""

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=config.GITHUB_ACTIVITY_LOOKBACK_DAYS)
    recent = 0
    for event in events:
        created = _parse_timestamp(event.get("created_at"))
        if created and created >= cutoff:
            recent += 1
    for threshold, points in config.GITHUB_ACTIVITY_THRESHOLDS:
        if recent >= threshold:
            return points
    return 0


def score_repos(repos: list[dict]) -> tuple[int, bool]:
    """Points (0-5) from non-fork owned repos, with a Python/AI relevance bonus."""

    owned = [r for r in repos if not r.get("fork", False)]
    count = len(owned)

    base = 0
    for threshold, points in config.GITHUB_REPO_COUNT_THRESHOLDS:
        if count >= threshold:
            base = points
            break

    relevant = any(_repo_is_relevant(r) for r in owned)
    bonus = config.GITHUB_RELEVANT_REPO_BONUS if relevant else 0
    return min(5, base + bonus), relevant


def _repo_is_relevant(repo: dict) -> bool:
    language = (repo.get("language") or "").lower()
    if language == "python":
        return True
    haystack = f"{repo.get('name', '')} {repo.get('description') or ''}".lower()
    if contains(haystack, "python"):
        return True
    return any(contains(haystack, term) for term in config.AI_KEYWORDS)


def _parse_timestamp(value: object) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class GithubEnricher:
    def __init__(self, settings: Optional[Settings] = None, transport: Optional[Transport] = None):
        self.settings = settings or Settings.from_env()
        self._transport = transport
        self._cache: dict[str, GithubEnrichment] = {}
        self._semaphore = asyncio.Semaphore(max(1, self.settings.github_max_concurrency))

    async def enrich(self, username: Optional[str]) -> GithubEnrichment:
        if not username:
            return GithubEnrichment(status="not_provided", summary="No GitHub profile provided")
        if username in self._cache:
            return self._cache[username]

        async with self._semaphore:
            result = await self._fetch_and_score(username)
        self._cache[username] = result
        return result

    async def enrich_many(self, usernames: list[Optional[str]]) -> dict[str, GithubEnrichment]:
        """Enrich a batch of usernames concurrently, de-duplicated."""

        unique = {u for u in usernames if u}
        results = await asyncio.gather(*(self.enrich(u) for u in unique))
        return dict(zip(unique, results))

    async def _fetch_and_score(self, username: str) -> GithubEnrichment:
        try:
            events_status, events = await self._get(f"/users/{username}/events/public")
            if events_status == 404:
                return GithubEnrichment(
                    status="not_found",
                    username=username,
                    summary=f"GitHub profile '{username}' not found",
                )
            if events_status == 403:
                return GithubEnrichment(
                    status="rate_limited",
                    username=username,
                    summary="GitHub API rate-limited; enrichment skipped",
                )
            if events_status != 200 or not isinstance(events, list):
                return GithubEnrichment(
                    status="error",
                    username=username,
                    summary=f"GitHub events request failed (HTTP {events_status})",
                )

            repos_status, repos = await self._get(f"/users/{username}/repos?per_page=100&sort=pushed")
            repos = repos if (repos_status == 200 and isinstance(repos, list)) else []

            activity = score_activity(events)
            repo_points, relevant = score_repos(repos)
            summary = (
                f"{len(events)} recent public events; {len(repos)} owned repos; "
                f"{'Python/AI repos present' if relevant else 'no clearly relevant repos'}"
            )
            return GithubEnrichment(
                status="ok",
                username=username,
                activity_points=activity,
                repo_points=repo_points,
                summary=summary,
            )
        except Exception as exc:  # noqa: BLE001 - enrichment must never break the batch
            logger.warning("GitHub enrichment failed for %s: %s", username, exc)
            return GithubEnrichment(
                status="error", username=username, summary=f"GitHub enrichment error: {exc}"
            )

    async def _get(self, path: str) -> tuple[int, object]:
        if self._transport is not None:
            return await self._transport(path)

        import httpx  # lazy import

        headers = {"Accept": "application/vnd.github+json"}
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        async with httpx.AsyncClient(timeout=config.GITHUB_REQUEST_TIMEOUT) as client:
            resp = await client.get(config.GITHUB_API_BASE + path, headers=headers)
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001 - non-JSON error bodies are tolerated
                data = None
            return resp.status_code, data
