from travel_rag.conversation import MAX_TURNS, ChatSession, Turn, format_history, rewrite_question
from travel_rag.rag import RAGPipeline
from travel_rag.retriever import SearchResult


class FakeLLM:
    """Records prompts; rewrites by a lookup table, answers with a fixed JSON."""

    def __init__(self):
        self.prompts = []

    def generate_json(self, prompt, system, schema):
        self.prompts.append(prompt)
        if "standalone_question" in schema["properties"]:
            return {"standalone_question": "Hội An có món ăn đặc sản nào?"}
        return {"found": True, "answer": "Cao lầu.", "citations": [1]}


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k=5):
        self.queries.append(query)
        chunk = {"chunk_id": "1-000", "title": "Hội An", "section": "Ẩm thực", "url": "u", "text": "Cao lầu."}
        return [SearchResult(rank=1, score=0.9, chunk=chunk)]


def test_first_turn_is_not_rewritten():
    llm = FakeLLM()
    assert rewrite_question("Hội An ở đâu?", [], llm) == "Hội An ở đâu?"
    assert llm.prompts == []


def test_history_is_truncated_to_recent_turns():
    history = [Turn(f"q{i}", f"a{i}") for i in range(MAX_TURNS + 2)]
    text = format_history(history)
    assert "q0" not in text and "q1" not in text
    assert f"q{MAX_TURNS + 1}" in text


def test_follow_up_is_rewritten_before_retrieval():
    llm, retriever = FakeLLM(), FakeRetriever()
    session = ChatSession(RAGPipeline(retriever, llm))
    session.ask("Hội An ở đâu?")
    out = session.ask("Còn món ăn ở đó thì sao?")
    assert retriever.queries == ["Hội An ở đâu?", "Hội An có món ăn đặc sản nào?"]
    assert out.rewritten_question == "Hội An có món ăn đặc sản nào?"
    assert [t.question for t in session.history] == ["Hội An ở đâu?", "Còn món ăn ở đó thì sao?"]
    assert out.rag.citations[0].chunk_id == "1-000"
