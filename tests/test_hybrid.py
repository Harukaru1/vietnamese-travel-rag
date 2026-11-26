from travel_rag.hybrid import RRF_K, rrf_fuse


def test_rrf_rewards_agreement_between_retrievers():
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "b", "d"]]))
    # b is 2nd in both lists, a and c are 1st in one list only
    assert fused["b"] == 2 / (RRF_K + 2)
    assert fused["c"] == 1 / (RRF_K + 1) + 1 / (RRF_K + 3)
    assert [cid for cid, _ in rrf_fuse([["a", "b", "c"], ["c", "b", "d"]])][:2] == ["c", "b"]


def test_rrf_keeps_items_found_by_one_retriever():
    fused = [cid for cid, _ in rrf_fuse([["a"], ["d"]])]
    assert set(fused) == {"a", "d"}


def test_parse_finetuned_output():
    from travel_rag.eval_finetuned import parse
    from travel_rag.rag import NOT_FOUND

    obj, answer, found, cites = parse('```json\n{"found": true, "answer": "Cao lầu.", "citations": [2]}\n```')
    assert obj and answer == "Cao lầu." and found and cites == [2]
    _, answer, found, _ = parse('{"found": false, "answer": "", "citations": []}')
    assert answer == NOT_FOUND and not found
    obj, answer, found, cites = parse("Hội An nổi tiếng với cao lầu.")
    assert obj is None and found and cites == []
