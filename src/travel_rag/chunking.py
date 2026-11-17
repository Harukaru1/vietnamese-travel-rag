"""Split Wikipedia articles into overlapping, section-aware chunks.

Pipeline for one article:

    raw text --parse_sections--> [(section title, paragraphs)]
             --split_sentences--> sentences per section
             --pack_sentences--> chunks of ~chunk_size words with ~overlap words shared

Design choices (each one is a knob we can evaluate on Day 5):

* Unit = whitespace-separated words. Vietnamese writes one syllable per
  "word", so this is a syllable count. It is simple, language-agnostic and
  roughly proportional to model tokens (the embedding tokenizer typically
  produces a bit more than one token per Vietnamese syllable).
* Chunks never cross a section boundary, so a citation points to one topic
  ("Ẩm thực", "Lịch sử", ...), and the section name is kept as metadata.
* Chunks end on sentence boundaries; only a sentence longer than chunk_size is
  cut mid-sentence.
* Overlap is made of whole sentences copied from the end of the previous chunk,
  so a fact that straddles a boundary still appears intact in one chunk.
* Reference-style sections (Tham khảo, Liên kết ngoài, ...) are dropped: they
  are lists of sources, not content.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Sections after which Vietnamese Wikipedia articles contain only references.
STOP_SECTIONS = {
    "tham khảo", "chú thích", "liên kết ngoài", "xem thêm", "ghi chú",
    "đọc thêm", "thư mục", "nguồn", "tài liệu tham khảo", "thư viện ảnh",
}

MAX_HEADING_CHARS = 60
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Chunk:
    chunk_id: str      # "<article id>-<index>", stable across runs with the same settings
    article_id: str
    title: str
    section: str       # "" for the lead section
    url: str
    category: str
    text: str
    n_words: int

    def embedding_text(self) -> str:
        """Text sent to the embedding model: title and section give context
        that the chunk itself often omits (e.g. 'Nó nằm ở ...')."""
        header = f"{self.title} — {self.section}" if self.section else self.title
        return f"{header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def looks_short(line: str) -> bool:
    return 0 < len(line) <= MAX_HEADING_CHARS and line[-1] not in ".!?…:;,)\"”"


def is_heading(line: str, prev_line: str | None) -> bool:
    """The dump has no markup, so headings are guessed: a short line without end
    punctuation that comes right after a normal paragraph. In a run of short
    lines ("Biến tấu", "Phở bò", "Phở gà", ...) only the first one is treated as
    the heading; the rest are list items and stay in the section text."""
    return looks_short(line) and prev_line is not None and not looks_short(prev_line)


def parse_sections(text: str) -> list[tuple[str, list[str]]]:
    """Return [(section title, [paragraph, ...]), ...] with reference sections removed."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sections: list[tuple[str, list[str]]] = [("", [])]
    for i, line in enumerate(lines):
        prev_line = lines[i - 1] if i > 0 else None
        if line.lower() in STOP_SECTIONS:
            break  # everything after "Tham khảo" etc. is references/navigation
        if is_heading(line, prev_line):
            sections.append((line, []))
        else:
            sections[-1][1].append(line)
    return [(title, paras) for title, paras in sections if paras]


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(paragraph) if s.strip()]


def n_words(text: str) -> int:
    return len(text.split())


def split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Hard-split a sentence that alone exceeds chunk_size words."""
    words = sentence.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def pack_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack sentences into chunks of at most chunk_size words.

    Each new chunk starts with the last sentences of the previous chunk whose
    total length is at most `overlap` words.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    units: list[str] = []
    for s in sentences:
        units.extend(split_long_sentence(s, chunk_size) if n_words(s) > chunk_size else [s])

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit_len = n_words(unit)
        if current and current_len + unit_len > chunk_size:
            chunks.append(" ".join(current))
            # Carry over trailing sentences as overlap.
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                prev_len = n_words(prev)
                if carry_len + prev_len > overlap or carry_len + prev_len + unit_len > chunk_size:
                    break
                carry.insert(0, prev)
                carry_len += prev_len
            current, current_len = carry, carry_len
        current.append(unit)
        current_len += unit_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_article(article: dict, chunk_size: int = 200, overlap: int = 40,
                  min_words: int = 20) -> list[Chunk]:
    """Chunk one article dict with keys id, title, url, category, text."""
    chunks: list[Chunk] = []
    for section, paragraphs in parse_sections(article["text"]):
        sentences = [s for p in paragraphs for s in split_sentences(p)]
        for text in pack_sentences(sentences, chunk_size, overlap):
            if n_words(text) < min_words:
                continue  # fragments such as a lone caption or "Nguyên văn Hán văn:"
            chunks.append(Chunk(
                chunk_id=f"{article['id']}-{len(chunks):03d}",
                article_id=str(article["id"]),
                title=article["title"],
                section=section,
                url=article["url"],
                category=article.get("category", ""),
                text=text,
                n_words=n_words(text),
            ))
    return chunks
