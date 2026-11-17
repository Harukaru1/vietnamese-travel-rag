"""Streamlit demo: ask a question about Vietnamese travel, get a cited answer.

    PYTHONPATH=src streamlit run app/streamlit_app.py

Needs a built index under index/ (see README). The index directory can be
chosen in the sidebar, so the fast e5-small index works too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from travel_rag.llm import DailyQuotaError, GeminiClient, LLMError  # noqa: E402
from travel_rag.rag import RAGPipeline, answer_closed_book  # noqa: E402
from travel_rag.retriever import INDEX_ROOT, Retriever  # noqa: E402

EXAMPLES = [
    "Phở có nguồn gốc từ đâu?",
    "Lễ hội chùa Hương diễn ra vào thời gian nào?",
    "Fansipan cao bao nhiêu mét?",
    "Giá vé tham quan Vịnh Hạ Long năm 2025 là bao nhiêu?",
]

st.set_page_config(page_title="Vietnamese Travel RAG", page_icon="🧭", layout="centered")


@st.cache_resource(show_spinner="Loading index and embedding model…")
def load_pipeline(index_name: str) -> RAGPipeline:
    return RAGPipeline(Retriever(INDEX_ROOT / index_name), GeminiClient())


def available_indexes() -> list[str]:
    if not INDEX_ROOT.exists():
        return []
    names = sorted(p.name for p in INDEX_ROOT.iterdir() if (p / "meta.json").exists())
    return sorted(names, key=lambda n: not n.startswith("bge-m3"))  # bge-m3 first


with st.sidebar:
    st.header("Settings")
    indexes = available_indexes()
    if not indexes:
        st.error("No index found. Build one first, see README.")
        st.stop()
    index_name = st.selectbox("Index (embedding model, chunk size)", indexes)
    k = st.slider("Passages given to the model (k)", 1, 10, 5)
    compare = st.checkbox("Also show the answer without retrieval", value=False)
    st.divider()
    st.caption(
        "Sources: Vietnamese Wikipedia, dump of 2023-11-01 (CC BY-SA 4.0), "
        "500 travel articles. Wikipedia is edited by volunteers and this snapshot "
        "predates the 2025 province mergers, so check important facts against "
        "official sources."
    )

st.title("Vietnamese Travel Q&A")
st.write("Ask about provinces, destinations, heritage sites, food or festivals. "
         "Answers come only from the retrieved Wikipedia passages and cite them.")

if "question" not in st.session_state:
    st.session_state.question = ""
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state.question = ex

question = st.text_input("Your question (Vietnamese)", key="question",
                         placeholder="Ví dụ: Món đặc sản nổi tiếng của Hội An là gì?")

if question.strip():
    pipeline = load_pipeline(index_name)
    with st.spinner("Searching…"):
        results = pipeline.retriever.search(question, k)
    try:
        with st.spinner("Writing the answer…"):
            answer = pipeline.answer(question, results=results)
    except DailyQuotaError:
        st.error("The Gemini free-tier daily quota is used up. The retrieved passages are shown below.")
        answer = None
    except LLMError as exc:
        st.error(f"The language model did not respond ({exc}). The retrieved passages are shown below.")
        answer = None

    if answer is not None:
        st.subheader("Answer")
        if answer.found:
            st.success(answer.answer)
        else:
            st.info(answer.answer + " The retrieved passages do not contain the answer.")
        if answer.citations:
            st.markdown("**Cited sources:** " + ", ".join(
                f"[{c.number}] [{c.title}{' › ' + c.section if c.section else ''}]({c.url})"
                for c in answer.citations))

    if compare:
        try:
            st.subheader("Without retrieval (for comparison)")
            st.warning(answer_closed_book(question, pipeline.llm) +
                       "  \n_No sources: this answer cannot be checked._")
        except LLMError as exc:
            st.error(f"Closed-book answer failed: {exc}")

    st.subheader("Retrieved passages")
    cited = {c.number for c in answer.citations} if answer else set()
    for i, r in enumerate(results, start=1):
        c = r.chunk
        where = f"{c['title']} › {c['section']}" if c["section"] else c["title"]
        mark = " ✅ cited" if i in cited else ""
        with st.expander(f"[{i}] {where} · similarity {r.score:.2f}{mark}", expanded=i in cited):
            st.write(c["text"])
            st.caption(f"[{c['url']}]({c['url']}) · chunk {c['chunk_id']}")
