import json

import numpy as np

from travel_rag.retriever import Retriever, build_faiss


class FakeEmbedder:
    """Maps a query to a fixed unit vector, so the test needs no model download."""

    def embed_queries(self, texts):
        table = {"north": [1.0, 0.0], "east": [0.0, 1.0], "northeast": [0.7071, 0.7071]}
        return np.array([table[t] for t in texts], dtype=np.float32)


def make_index(tmp_path):
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    import faiss
    faiss.write_index(build_faiss(vectors), str(tmp_path / "index.faiss"))
    chunks = [{"chunk_id": f"c{i}"} for i in range(3)]
    (tmp_path / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks))
    (tmp_path / "meta.json").write_text(json.dumps({"model_name": "fake", "max_seq_length": 8}))
    return Retriever(tmp_path, embedder=FakeEmbedder())


def test_search_ranks_by_cosine(tmp_path):
    r = make_index(tmp_path)
    res = r.search("north", k=3)
    assert [x.chunk["chunk_id"] for x in res] == ["c0", "c2", "c1"]
    assert np.isclose(res[0].score, 1.0) and np.isclose(res[1].score, 0.6)
    assert [x.rank for x in res] == [1, 2, 3]


def test_search_many_and_k(tmp_path):
    r = make_index(tmp_path)
    res = r.search_many(["east", "northeast"], k=1)
    assert [x[0].chunk["chunk_id"] for x in res] == ["c1", "c2"]
