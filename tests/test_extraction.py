from resume_screener.extraction import _guess_name, extract_candidate
from resume_screener.llm_adapter import LLMClient
from resume_screener.parsing import ParsedResume


def _parsed(text, email="jane@example.com", user="janedoe"):
    return ParsedResume(
        source_file="candidate_01.pdf",
        text=text,
        email=email,
        github_url=f"https://github.com/{user}" if user else None,
        github_username=user,
    )


def test_success_merges_llm_and_regex_fields():
    def transport(system, user_prompt, schema):
        return {
            "name": "Jane Doe",
            "skills": ["Python", "LangChain", "FastAPI"],
            "projects": [
                {
                    "name": "RAG assistant",
                    "description": "Retrieval augmented chatbot",
                    "technologies": ["Python", "LangChain"],
                }
            ],
            "work_experience": [],
        }

    client = LLMClient(transport=transport)
    parsed = _parsed("Jane Doe\nPython LangChain FastAPI")
    result = extract_candidate(parsed, client)

    assert result.extraction_status == "ok"
    assert result.name == "Jane Doe"
    assert result.email == "jane@example.com"  # regex-sourced
    assert result.github_username == "janedoe"
    assert result.projects[0].name == "RAG assistant"
    assert "LangChain" in result.skills


def test_llm_failure_falls_back_to_heuristics():
    def transport(system, user_prompt, schema):
        raise ValueError("model down")

    client = LLMClient(transport=transport)
    parsed = _parsed("Bob Lee\nExperienced in Python, FastAPI, Docker and LangChain")
    result = extract_candidate(parsed, client)

    assert result.extraction_status == "llm_failed"
    # Heuristic keyword scan should still surface known technologies.
    assert "python" in result.skills
    assert "langchain" in result.skills
    assert "docker" in result.skills
    # Atomic fields still populated from regex.
    assert result.email == "jane@example.com"
    assert result.github_username == "janedoe"


def test_name_falls_back_when_llm_omits_it():
    def transport(system, user_prompt, schema):
        return {"name": None, "skills": ["Python"], "projects": [], "work_experience": []}

    client = LLMClient(transport=transport)
    parsed = _parsed("Asha Rao\nSoftware Engineer\nPython")
    result = extract_candidate(parsed, client)
    assert result.name == "Asha Rao"


def test_guess_name_skips_contact_lines():
    text = "jane@example.com\nhttps://github.com/janedoe\n2024\nJane Doe\nPython"
    assert _guess_name(text) == "Jane Doe"


def test_guess_name_none_when_no_plausible_line():
    assert _guess_name("me@x.com\n12345") is None
