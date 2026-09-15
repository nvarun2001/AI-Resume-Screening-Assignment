from resume_screener.eligibility import evaluate_eligibility
from resume_screener.models import Project, ResumeExtraction


def _extraction(skills=None, projects=None, raw_text=""):
    return ResumeExtraction(
        source_file="c.pdf",
        skills=skills or [],
        projects=projects or [],
        raw_text=raw_text,
    )


def test_python_and_ai_passes():
    extraction = _extraction(
        skills=["Python", "LangChain"],
        projects=[Project(name="RAG bot", technologies=["Python", "LangChain"])],
    )
    result = evaluate_eligibility(extraction)
    assert result.eligible is True
    assert result.rejection_reasons == []


def test_python_only_rejected():
    extraction = _extraction(skills=["Python", "FastAPI", "PostgreSQL"])
    result = evaluate_eligibility(extraction)
    assert result.eligible is False
    assert any("AI" in r for r in result.rejection_reasons)


def test_ai_only_rejected():
    extraction = _extraction(skills=["JavaScript", "LangChain", "React"])
    result = evaluate_eligibility(extraction)
    assert result.eligible is False
    assert any("Python" in r for r in result.rejection_reasons)


def test_java_react_only_rejected_with_both_reasons():
    extraction = _extraction(skills=["Java", "React", "Spring Boot"])
    result = evaluate_eligibility(extraction)
    assert result.eligible is False
    assert len(result.rejection_reasons) == 2
    assert "React" in result.matched_skills


def test_js_plus_python_and_ai_still_passes():
    # JavaScript/React presence must not disqualify a Python+AI candidate.
    extraction = _extraction(
        skills=["JavaScript", "React", "Next.js", "Python"],
        raw_text="Built a RAG pipeline with embeddings and Python.",
    )
    result = evaluate_eligibility(extraction)
    assert result.eligible is True


def test_python_framework_implies_python():
    extraction = _extraction(
        skills=["Django", "LangGraph"],
        raw_text="Agentic workflow built with Django and LangGraph.",
    )
    result = evaluate_eligibility(extraction)
    assert result.eligible is True
    assert result.python_evidence == "django"


def test_evidence_found_in_raw_text_only():
    extraction = _extraction(
        skills=[],
        raw_text="Implemented a multi-agent system in python using vector search.",
    )
    result = evaluate_eligibility(extraction)
    assert result.eligible is True


def test_empty_extraction_rejected():
    result = evaluate_eligibility(_extraction())
    assert result.eligible is False
    assert len(result.rejection_reasons) == 2


def test_rag_does_not_match_storage():
    # Guard against substring false positives: 'storage' must not trigger 'rag'.
    extraction = _extraction(skills=["Python"], raw_text="Used cloud storage and average latency")
    result = evaluate_eligibility(extraction)
    assert result.eligible is False
    assert result.ai_evidence is None
