"""Streamlit chat demo: multi-turn Vietnamese travel Q&A with cited answers.

    PYTHONPATH=src streamlit run app/streamlit_app.py

Needs a built index under index/ (see README). The index can be chosen in the
sidebar, so the fast e5-small index works too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from travel_rag.conversation import ChatSession  # noqa: E402
from travel_rag.llm import DailyQuotaError, GeminiClient, LLMError  # noqa: E402
from travel_rag.rag import RAGPipeline, answer_closed_book  # noqa: E402
from travel_rag.retriever import INDEX_ROOT, Retriever  # noqa: E402

EXAMPLES = [
    "Hội An có gì nổi tiếng?",
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


def render_assistant(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["error"])
    elif msg["found"]:
        st.markdown(msg["answer"])
    else:
        st.info(msg["answer"] + " The retrieved passages do not contain the answer.")
    if msg.get("rewritten") and msg["rewritten"] != msg["question"]:
        st.caption(f"Understood as: _{msg['rewritten']}_")
    if msg.get("citations"):
        st.markdown("**Sources:** " + ", ".join(
            f"[{c['number']}] [{c['where']}]({c['url']})" for c in msg["citations"]))
    if msg.get("closed_book"):
        st.warning(f"Without retrieval: {msg['closed_book']}  \n_No sources: this answer cannot be checked._")
    if msg.get("passages"):
        with st.expander(f"Retrieved passages ({len(msg['passages'])})"):
            for p in msg["passages"]:
                st.markdown(f"**[{p['n']}] {p['where']}** · similarity {p['score']:.2f}"
                            + (" · ✅ cited" if p["cited"] else ""))
                st.write(p["text"])
                st.caption(f"[{p['url']}]({p['url']})")


with st.sidebar:
    st.header("Settings")
    indexes = available_indexes()
    if not indexes:
        st.error("No index found. Build one first, see README.")
        st.stop()
    index_name = st.selectbox("Index (embedding model, chunk size)", indexes)
    k = st.slider("Passages given to the model (k)", 1, 10, 5)
    compare = st.checkbox("Also show the answer without retrieval", value=False)
    if st.button("New conversation", use_container_width=True):
        st.session_state.pop("session", None)
        st.session_state.pop("messages", None)
    st.divider()
    st.caption(
        "Sources: Vietnamese Wikipedia, dump of 2023-11-01 (CC BY-SA 4.0), "
        "500 travel articles. Wikipedia is edited by volunteers and this snapshot "
        "predates the 2025 province mergers, so check important facts against "
        "official sources."
    )

pipeline = load_pipeline(index_name)
if "session" not in st.session_state or st.session_state.get("index") != index_name:
    st.session_state.session = ChatSession(pipeline)
    st.session_state.messages = []
    st.session_state.index = index_name

st.title("Vietnamese Travel Q&A")
st.write("Ask about provinces, destinations, heritage sites, food or festivals, and "
         "follow up as in a chat. Answers come only from retrieved Wikipedia passages.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["question"])
        else:
            render_assistant(msg)

clicked = None
if not st.session_state.messages:
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, use_container_width=True):
            clicked = ex

question = st.chat_input("Hỏi về du lịch Việt Nam… (ví dụ: Còn món ăn ở đó thì sao?)") or clicked
if question:
    st.session_state.messages.append({"role": "user", "question": question})
    with st.chat_message("user"):
        st.markdown(question)
    msg = {"role": "assistant", "question": question}
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and writing the answer…"):
                out = st.session_state.session.ask(question, k=k)
            rag = out.rag
            cited = {c.number for c in rag.citations}
            msg.update(
                answer=rag.answer, found=rag.found, rewritten=out.rewritten_question,
                citations=[{"number": c.number, "url": c.url,
                            "where": f"{c.title} › {c.section}" if c.section else c.title}
                           for c in rag.citations],
                passages=[{"n": i, "score": r.score, "cited": i in cited, "text": r.chunk["text"],
                           "url": r.chunk["url"],
                           "where": f"{r.chunk['title']} › {r.chunk['section']}" if r.chunk["section"]
                           else r.chunk["title"]}
                          for i, r in enumerate(rag.retrieved, start=1)],
            )
            if compare:
                msg["closed_book"] = answer_closed_book(out.rewritten_question, pipeline.llm)
        except DailyQuotaError:
            msg["error"] = "The Gemini free-tier daily quota is used up. Please try again later."
        except LLMError as exc:
            msg["error"] = f"The language model did not respond ({exc})."
        render_assistant(msg)
    st.session_state.messages.append(msg)
