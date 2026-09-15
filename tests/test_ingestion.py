from pathlib import Path

from resume_screener.ingestion import discover_resumes


def _write(path: Path, content: str = "content") -> None:
    path.write_text(content, encoding="utf-8")


def test_discovers_supported_extensions(tmp_path):
    _write(tmp_path / "candidate_01.pdf")
    _write(tmp_path / "candidate_02.docx")
    _write(tmp_path / "candidate_03.txt")
    found = discover_resumes(tmp_path)
    assert len(found) == 3


def test_ignores_unsupported_files(tmp_path):
    _write(tmp_path / "resume.pdf")
    _write(tmp_path / "notes.md")
    _write(tmp_path / "photo.png")
    found = discover_resumes(tmp_path)
    names = [p.name for p in found]
    assert names == ["resume.pdf"]


def test_skips_empty_files(tmp_path):
    _write(tmp_path / "good.pdf", "text")
    (tmp_path / "empty.pdf").write_text("", encoding="utf-8")
    found = discover_resumes(tmp_path)
    assert [p.name for p in found] == ["good.pdf"]


def test_skips_hidden_files(tmp_path):
    _write(tmp_path / ".hidden.pdf")
    _write(tmp_path / "visible.pdf")
    found = discover_resumes(tmp_path)
    assert [p.name for p in found] == ["visible.pdf"]


def test_recurses_into_subdirectories(tmp_path):
    sub = tmp_path / "batch1"
    sub.mkdir()
    _write(sub / "candidate.pdf")
    _write(tmp_path / "top.pdf")
    found = discover_resumes(tmp_path)
    assert len(found) == 2


def test_missing_directory_returns_empty(tmp_path):
    assert discover_resumes(tmp_path / "does_not_exist") == []


def test_case_insensitive_extension(tmp_path):
    _write(tmp_path / "resume.PDF")
    found = discover_resumes(tmp_path)
    assert len(found) == 1
