from travel_rag.rag import NOT_FOUND, build_prompt, parse_answer
from travel_rag.retriever import SearchResult


def results():
    chunks = [
        {"chunk_id": "1-000", "title": "Phở", "section": "", "url": "u1", "text": "Phở là món ăn."},
        {"chunk_id": "1-001", "title": "Phở", "section": "Nguồn gốc", "url": "u1", "text": "Phở ra đời ở Nam Định."},
    ]
    return [SearchResult(rank=i + 1, score=0.9, chunk=c) for i, c in enumerate(chunks)]


def test_prompt_numbers_sources_with_section():
    prompt = build_prompt("Phở từ đâu?", results())
    assert "[1] Phở\nPhở là món ăn." in prompt
    assert "[2] Phở — Nguồn gốc\n" in prompt
    assert prompt.endswith("Question: Phở từ đâu?")


def test_valid_citations_map_to_chunks_and_invalid_are_dropped():
    out = {"found": True, "answer": " Nam Định. ", "citations": [2, 2, 7, 0]}
    a = parse_answer("q", out, results())
    assert a.answer == "Nam Định."
    assert [c.chunk_id for c in a.citations] == ["1-001"]
    assert a.invalid_citations == [7, 0]


def test_not_found_returns_fixed_string_without_citations():
    for out in ({"found": False, "answer": "", "citations": [1]},
                {"found": True, "answer": "  ", "citations": [1]}):
        a = parse_answer("q", out, results())
        assert not a.found and a.answer == NOT_FOUND and a.citations == []
