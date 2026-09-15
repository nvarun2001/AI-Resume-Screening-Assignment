import json
import sys
from pathlib import Path

import main as cli

ELIGIBLE = """Asha Rao
Built a multi-agent RAG pipeline in Python with LangGraph, retrieval and
tool calling. FastAPI backend with PostgreSQL. Deployed on GCP with Docker.
asha@example.com
"""

REJECTED = """Bob Jones
Java, React, Spring Boot developer.
bob@example.com
"""


def test_cli_writes_valid_json(tmp_path):
    resumes = tmp_path / "resumes"
    resumes.mkdir()
    (resumes / "candidate_01.txt").write_text(ELIGIBLE, encoding="utf-8")
    (resumes / "candidate_02.txt").write_text(REJECTED, encoding="utf-8")
    output = tmp_path / "out" / "results.json"

    # --no-llm keeps the run offline (no LLM, and no GitHub links in fixtures).
    exit_code = cli.main(["--input", str(resumes), "--output", str(output), "--no-llm"])
    assert exit_code == 0
    assert output.exists()

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["summary"]["total_resumes"] == 2
    assert data["summary"]["eligible"] == 1
    assert data["summary"]["rejected"] == 1

    candidates = data["candidates"]
    top = candidates[0]
    assert top["rank"] == 1
    assert top["eligible"] is True
    assert "score_breakdown" in top
    assert set(top["score_breakdown"]) >= {
        "ai_project_depth",
        "python_backend",
        "cloud_fullstack",
        "github",
        "engineering_depth",
    }


def test_cli_missing_input_arg_exits():
    try:
        cli.parse_args([])
        assert False, "expected SystemExit"
    except SystemExit:
        pass
