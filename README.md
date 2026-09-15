# AI-Resume-Screening-Assignment

A small, production-minded pipeline that screens a folder of resumes for an SDE
internship requiring strong Python fundamentals and practical AI/agentic
experience. It ingests resumes, applies a hard eligibility filter, scores
eligible candidates on a 100-point rubric, enriches scores with public GitHub
activity, and writes a ranked, evidence-backed `results.json`.

The design goal is a correct, explainable backend: every eligibility decision
lists its reasons and every score carries evidence.

## Pipeline at a glance

```
ingest → parse (text + email/GitHub) → extract (LLM) → eligibility (rules)
       → score (hybrid) → GitHub enrichment → rank → results.json
```

1. **Ingest** every `.pdf` (and optionally `.docx` / `.txt`) in the input folder.
2. **Parse** text with pdfplumber/python-docx; pull email and GitHub username via regex.
3. **Extract** skills, projects, and experience into a structured schema using the LLM.
4. **Filter** on hard, rule-based eligibility (Python **and** AI evidence).
5. **Score** eligible candidates across five weighted categories.
6. **Enrich** with public GitHub activity (best-effort, capped at 10 points).
7. **Rank** eligible candidates and emit a batch summary.

## Requirements

- Python 3.11+ (developed on 3.14)
- An Anthropic API key (or any OpenAI-compatible endpoint). The pipeline also
  runs without an LLM in a deterministic `--no-llm` mode.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then fill in your key
```

`.env` values:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `anthropic` (native Claude) or `openai` (OpenAI-compatible) |
| `LLM_MODEL` | e.g. `claude-sonnet-4-6` |
| `LLM_API_KEY` | your provider API key |
| `LLM_BASE_URL` | optional, only for OpenAI-compatible endpoints |
| `GITHUB_TOKEN` | optional; raises GitHub rate limit from 60/hr to 5000/hr |
| `LLM_MAX_CONCURRENCY` / `GITHUB_MAX_CONCURRENCY` | bounded concurrency knobs |

Secrets are read only from the environment and never committed (`.env` is
gitignored).

## Usage

```bash
python main.py --input ./resumes --output ./output/results.json
```

Options:

- `--no-llm` — skip the LLM and score deterministically (useful for a quick,
  offline run or when no key is available).
- `-v` / `--verbose` — enable info-level logging.

## Generating `results.json`

Run the pipeline against the resume folder and write the ranked output. With a
valid `LLM_API_KEY` in `.env`, this uses the LLM for extraction and AI-depth
scoring:

```bash
python main.py --input ./resumes --output ./output/results.json
```

No API key or credits? Use the deterministic mode — it produces a complete,
ranked `results.json` from keyword-based heuristics (no network/LLM calls):

```bash
python main.py --input ./resumes --output ./output/results.json --no-llm
```

## Output

A single JSON object with a batch summary and the ranked candidate list:

```json
{
  "summary": {
    "total_resumes": 50,
    "successfully_parsed": 49,
    "eligible": 21,
    "rejected": 28,
    "failed": 1
  },
  "candidates": [
    {
      "rank": 1,
      "candidate_name": "Asha Rao",
      "eligible": true,
      "total_score": 86,
      "score_breakdown": {
        "ai_project_depth": 35,
        "python_backend": 27,
        "cloud_fullstack": 12,
        "github": 8,
        "engineering_depth": 4
      },
      "matched_skills": ["Python", "FastAPI", "PostgreSQL", "LangGraph", "Docker", "GCP"],
      "project_summary": "Built a stateful agentic workflow with retrieval and tool calling.",
      "github_summary": "12 recent public events; 6 owned repos; Python/AI repos present",
      "strengths": ["Strong agentic project", "Async FastAPI backend"],
      "concerns": ["Limited Redis evidence"]
    }
  ]
}
```

Rejected candidates include `eligible: false`, `rejection_reasons`, and their
`matched_skills`. Unreadable resumes are recorded with `parse_status: "failed"`
so a single bad file never aborts the batch.

## Scoring rubric (100 points)

| Category | Weight | What it rewards |
|---|---|---|
| AI / Agentic / RAG project depth | 40 | Real agents, RAG, retrieval, tools, state, orchestration, evaluation |
| Python & backend engineering | 30 | Python, FastAPI, async, PostgreSQL, Redis — evidence over keywords |
| Cloud / deployment / full stack | 15 | GCP, Docker, deployment; React/Next as supporting signals |
| GitHub activity | 10 | Recent public activity and maintained/relevant repos |
| Engineering depth signals | 5 | Testing, caching, queues, observability, concurrency |

## Design decisions

**Filtering strategy.** Eligibility is a hard, deterministic, rule-based gate
that runs *outside* the LLM. It operates on the extracted structured fields plus
the raw text and requires evidence of **both** Python (as a skill, project
technology, or a Python-specific framework such as Django/Flask/FastAPI) **and**
a meaningful AI/LLM/RAG/agentic signal. Matching uses non-alphanumeric
boundaries so short tokens like `rag` don't match inside `storage`. JavaScript,
Java, or React never disqualify a candidate as long as the Python + AI bar is
met. Keeping this gate rule-based makes it predictable, unit-testable, and easy
to explain — a well-written but irrelevant resume can't talk its way in.

**Scoring strategy.** Scoring is a hybrid. Python/backend, cloud/full-stack, and
engineering-depth are scored deterministically from keyword evidence, weighting
matches found in a project or work description above skills-list-only mentions
(keyword-only matches are discounted). AI project depth — the largest and most
judgment-heavy category — is assessed by the LLM, which returns a structured
verdict with base points, a wrapper/tutorial penalty (5–15 points), and
evidence. A deterministic fallback covers the case where the LLM is unavailable
or fails for a given resume, so the batch always completes.

**LLM usage.** The LLM is used for two things only: turning messy resume text
into a structured schema (skills/projects/experience), and judging AI project
depth. Both use structured output (Pydantic schemas) — via Anthropic tool-use or
OpenAI JSON mode — so results are parseable and auditable. All provider
specifics sit behind a small adapter (`llm_adapter.py`) so the provider can be
swapped by changing `.env`. Every call retries once and then degrades to
heuristics rather than failing the batch. Hard eligibility is deliberately kept
out of the LLM.

**GitHub scoring.** If a resume contains a GitHub URL, the first path segment is
treated as the username and two public endpoints are queried: recent public
events (0–5 points for activity in the last 90 days) and owned repositories (0–5
points for non-fork repos, with a bonus when any repo is Python/AI relevant),
capped at 10. A missing, private, not-found, or rate-limited profile is recorded
as a status with zero points and never fails screening. Results are cached per
run and lookups are concurrency-bounded.

## Reliability

- Each resume is parsed, extracted, filtered, and scored independently; a single
  malformed file, LLM error, or GitHub failure degrades only that candidate.
- External calls (PDF parse, LLM, GitHub) are individually guarded.
- Configuration (weights, thresholds, keyword vocabularies, model, keys) is
  separated from business logic in `config.py`.
- GitHub lookups run concurrently with bounded concurrency; results are cached.

## Testing

```bash
pytest
```

The suite covers ingestion, parsing (including a real generated PDF and a
corrupt file), the LLM adapter (mocked — no network), extraction and its
fallback, eligibility edge cases, scoring and the wrapper penalty, GitHub
enrichment (success/404/rate-limit/error, mocked), ranking, and an end-to-end
pipeline run. No test makes a network or paid API call.

## Project structure

```
Assignment/
  resumes/                     input resumes (gitignored; not committed)
  output/results.json          generated result
  src/resume_screener/
    ingestion.py               find resume files
    parsing.py                 text extraction + email/GitHub regex
    llm_adapter.py             provider-agnostic structured-output client
    extraction.py              LLM extraction + heuristic fallback
    eligibility.py             deterministic Python + AI gate
    scoring.py                 hybrid 100-point scoring
    github_enrichment.py       public GitHub activity signal
    ranking.py                 ranking + batch summary
    pipeline.py                orchestration
    models.py                  Pydantic models
    config.py                  weights, keywords, settings
  tests/
  main.py                      CLI entrypoint
```

## If I had more time

- **Verify GitHub username against candidate context.** A `github.com/{org}/{repo}`
  link may point at an employer or course org rather than the candidate; I'd
  cross-check the username against the candidate's name/context before scoring.
- **OCR fallback for scanned PDFs.** Image-only resumes currently yield no text
  and are marked failed; a Tesseract fallback would recover them.
- **Bounded-concurrency LLM calls.** LLM extraction/scoring run sequentially for
  simplicity; moving them to a bounded async/worker pool would speed up large
  batches (GitHub I/O is already concurrent).
- **Richer GitHub signal with caching to disk.** Look at languages, recent
  commit cadence, and README quality, and persist a small on-disk cache across
  runs to avoid repeat calls and rate limits.
```
