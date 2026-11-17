from travel_rag.chunking import chunk_article, n_words, pack_sentences, parse_sections

ARTICLE_TEXT = """Phở là một món ăn truyền thống của Việt Nam. Thành phần chính là bánh phở và nước dùng.

Nguồn gốc
Phở ra đời vào đầu thế kỷ 20. Nam Định và Hà Nội thường được coi là quê hương của phở.

Biến tấu
Phở bò
Phở gà
Phở có nhiều biến thể ở các vùng miền khác nhau của đất nước.

Tham khảo
Một nguồn tham khảo nào đó.
"""


def test_parse_sections_drops_references_and_keeps_list_items():
    sections = parse_sections(ARTICLE_TEXT)
    titles = [t for t, _ in sections]
    assert titles == ["", "Nguồn gốc", "Biến tấu"]
    # list items are not mistaken for headings
    assert sections[2][1][:2] == ["Phở bò", "Phở gà"]
    assert all("tham khảo" not in p.lower() for _, ps in sections for p in ps)


def test_pack_respects_size_and_overlap():
    sentences = [f"Câu {i} có đúng sáu từ." for i in range(20)]  # 6 words each
    chunks = pack_sentences(sentences, chunk_size=30, overlap=12)
    assert all(n_words(c) <= 30 for c in chunks)
    # 5 sentences fit in 30 words; the next chunk repeats the last 2 (= 12 words)
    assert chunks[0] == " ".join(sentences[0:5])
    assert chunks[1] == " ".join(sentences[3:8])
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_zero_overlap_partitions_text():
    sentences = [f"Câu {i} gồm năm từ." for i in range(10)]  # 5 words each
    chunks = pack_sentences(sentences, chunk_size=10, overlap=0)
    assert chunks == [" ".join(sentences[i:i + 2]) for i in range(0, 10, 2)]


def test_long_sentence_is_hard_split():
    long_sentence = " ".join(["từ"] * 25)
    chunks = pack_sentences([long_sentence], chunk_size=10, overlap=0)
    assert [n_words(c) for c in chunks] == [10, 10, 5]


def test_chunk_article_metadata():
    article = {"id": "42", "title": "Phở", "url": "https://vi.wikipedia.org/wiki/Ph%E1%BB%9F",
               "category": "food", "text": ARTICLE_TEXT}
    chunks = chunk_article(article, chunk_size=50, overlap=10, min_words=5)
    assert [c.chunk_id for c in chunks] == ["42-000", "42-001", "42-002"]
    assert [c.section for c in chunks] == ["", "Nguồn gốc", "Biến tấu"]
    assert chunks[1].embedding_text().startswith("Phở — Nguồn gốc\n")
