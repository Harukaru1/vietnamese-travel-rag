"""Select ~500 travel-related articles from Vietnamese Wikipedia.

The dump has no category metadata (only id, url, title, text), so selection is
rule-based and transparent:

1. Candidate titles: the 63 provinces/municipalities, a hand-curated list of
   well-known destinations / heritage sites / dishes / festivals, and titles that
   start with a category prefix ("Vịnh ", "Chùa ", "Bánh ", "Lễ hội ", ...).
2. Filters on the lead section: must mention Vietnam (or a province), must not
   look like a biography or a list page, and must contain a category keyword.
3. Ranking inside each category: curated titles first, then longer articles
   (a rough proxy for content quality, since many short pages are bot stubs).
4. Per-category quotas keep the corpus balanced.

Note: the dump is from 2023-11-01, before the 2025 province mergers, so it
describes the 63-province administrative map.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

from travel_rag.config import ARTICLES_PATH, PROCESSED_DIR, RAW_DIR

PROVINCES = [
    "Hà Nội", "Thành phố Hồ Chí Minh", "Hải Phòng", "Đà Nẵng", "Cần Thơ",
    "An Giang", "Bà Rịa – Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu",
    "Bắc Ninh", "Bến Tre", "Bình Định", "Bình Dương", "Bình Phước",
    "Bình Thuận", "Cà Mau", "Cao Bằng", "Đắk Lắk", "Đắk Nông", "Điện Biên",
    "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Tĩnh",
    "Hải Dương", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa",
    "Kiên Giang", "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai",
    "Long An", "Nam Định", "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ",
    "Phú Yên", "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh",
    "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình",
    "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang", "Trà Vinh",
    "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái",
]

# Hand-picked titles that are central to Vietnamese travel but would not be
# caught by the prefix rules. Titles missing from the dump are simply skipped.
CURATED = {
    "destination": [
        "Hội An", "Sa Pa", "Đà Lạt", "Phú Quốc", "Côn Đảo", "Nha Trang",
        "Vũng Tàu", "Mũi Né", "Quần đảo Cát Bà", "Vườn quốc gia Cát Bà", "Huế", "Hạ Long", "Mộc Châu",
        "Mù Cang Chải", "Đồng Văn", "Cao nguyên đá Đồng Văn", "Bà Nà", "Fansipan",
        "Phan Thiết", "Quy Nhơn", "Lý Sơn", "Tam Đảo", "Hang Sơn Đoòng", "Phố cổ Đồng Văn",
        "Vịnh Hạ Long", "Vườn quốc gia Phong Nha – Kẻ Bàng", "Tam Cốc – Bích Động",
        "Đồng bằng sông Cửu Long", "Mỹ Tho", "Châu Đốc", "Buôn Ma Thuột", "Pleiku",
        "Hồ Ba Bể", "Thác Bản Giốc", "Núi Bà Đen", "Đèo Hải Vân", "Đèo Mã Pí Lèng",
        "Hồ Tây", "Hồ Hoàn Kiếm", "Vịnh Lan Hạ", "Bãi Cháy", "Chợ nổi Cái Răng",
    ],
    "heritage": [
        "Quần thể di tích Cố đô Huế", "Thánh địa Mỹ Sơn", "Quần thể danh thắng Tràng An",
        "Hoàng thành Thăng Long", "Thành nhà Hồ", "Văn Miếu – Quốc Tử Giám",
        "Địa đạo Củ Chi", "Dinh Độc Lập", "Nhà thờ chính tòa Đức Bà Sài Gòn",
        "Chợ Bến Thành", "Lăng Chủ tịch Hồ Chí Minh", "Chùa Thiên Mụ",
        "Chùa Bái Đính", "Chùa Hương", "Phố cổ Hội An", "Miếu Bà Chúa Xứ Núi Sam",
        "Chùa Một Cột", "Chùa Trấn Quốc", "Chùa Cầu", "Lăng Tự Đức", "Thiền viện Trúc Lâm Yên Tử",
        "Lăng Khải Định", "Lăng Minh Mạng", "Quần thể di tích Cố đô Hoa Lư", "Kinh thành Huế",
        "Nhà hát Lớn Hà Nội", "Cầu Long Biên", "Cầu Rồng", "Tháp Po Nagar", "Phủ Dầy",
        "Yên Tử", "Đền Hùng", "Nhà tù Hỏa Lò", "Bảo tàng Chứng tích chiến tranh",
        "Nhã nhạc cung đình Huế", "Cồng chiêng",
        "Ca trù", "Quan họ", "Đờn ca tài tử Nam Bộ", "Hát xoan", "Múa rối nước",
    ],
    "food": [
        "Phở", "Bánh mì", "Bún chả", "Bún bò Huế", "Cao lầu", "Mì Quảng",
        "Bánh xèo", "Gỏi cuốn", "Cơm tấm", "Bánh chưng", "Bánh tét",
        "Bánh cuốn", "Bún riêu cua", "Hủ tiếu", "Cà phê trứng", "Chả cá Lã Vọng",
        "Nước mắm", "Ẩm thực Việt Nam", "Ẩm thực Huế", "Cà phê vối",
        "Bánh bèo", "Bánh căn", "Nem chua", "Nem rán", "Xôi", "Cháo",
    ],
    "festival": [
        "Tết Nguyên Đán", "Tết Trung thu", "Tết Đoan ngọ", "Tết Hàn thực",
        "Giỗ Tổ Hùng Vương", "Tết Nguyên tiêu", "Táo quân", "Thanh minh",
        "Lễ hội chùa Hương", "Hội Gióng", "Hội Lim", "Lễ hội Cồng chiêng",
        "Festival Huế", "Lễ hội pháo hoa quốc tế Đà Nẵng", "Lễ hội miếu Bà Chúa Xứ",
        "Khâu Vai", "Lễ hội Nghinh Ông Quan Thánh Đế Quân", "Lễ hội tháp Bà Po Nagar",
        "Lễ hội Hoa Lư", "Lễ hội cà phê Buôn Ma Thuột", "Lễ hội chọi trâu Đồ Sơn",
    ],
}

PREFIXES = {
    "destination": [
        "Vịnh ", "Vườn quốc gia ", "Đảo ", "Hồ ", "Động ", "Thác ", "Bãi biển",
        "Khu bảo tồn", "Đèo ", "Núi ", "Ruộng bậc thang", "Hang ", "Mũi ",
        "Quần đảo ", "Cù lao ",
    ],
    "heritage": [
        "Chùa ", "Đền ", "Đình ", "Lăng ", "Thành cổ", "Di tích", "Phố cổ",
        "Nhà thờ ", "Bảo tàng ", "Chợ ", "Tháp ", "Thánh địa", "Quần thể",
        "Hoàng thành", "Kinh thành", "Văn Miếu", "Dinh ",
    ],
    "food": [
        "Bánh ", "Bún ", "Phở ", "Chè ", "Gỏi ", "Nem ", "Cơm ", "Mì ", "Lẩu ",
        "Canh ", "Cháo ", "Xôi ", "Chả ", "Hủ tiếu", "Cà phê ", "Rượu ", "Mắm ",
    ],
    "festival": ["Lễ hội ", "Hội ", "Tết "],
}

# Prefix matches that are not travel content (coups, earthquakes, Chinese
# provinces, organisations, ...). Checked before the prefix rules.
EXCLUDED_PREFIXES = [
    "Đảo chính", "Động đất", "Động cơ", "Động vật", "Động lực", "Hồ Bắc", "Hồ Nam",
    "Hồ cầm", "Núi lửa", "Hội Thánh", "Hội Tam", "Hội trường", "Hội họa", "Hội Doanh",
    "Hội đồng", "Hội nghị", "Hội Liên hiệp", "Hội chứng", "Hội quán", "Đình chỉ",
    "Đình Lập", "Tết Trung Quốc", "Mì ăn liền", "Mì sợi", "Chợ tại", "Chùa Việt Nam",
    "Lễ hội Việt Nam", "Tháp chùa", "Di tích ở", "Di tích về", "Di tích Việt Nam",
    "Mũi tên", "Hồ sơ", "Quần thể (", "Dinh điền", "Tết Nhật Bản", "Hang động Việt Nam",
    # found by manually reading the selected list (films, a surname, a person, ...)
    "Đảo của dân ngụ cư", "Đảo độc đắc", "Hồ (họ)", "Hồ Hán Dân", "Chợ nông sản", "Hội trăng",
]

# Lead-section keywords required for prefix-matched candidates of each category.
CATEGORY_KEYWORDS = {
    "destination": ["du lịch", "thắng cảnh", "danh thắng", "khu bảo tồn", "vườn quốc gia",
                    "đảo", "vịnh", "hồ", "thác", "hang", "núi", "bãi biển"],
    "heritage": ["di tích", "di sản", "chùa", "đền", "đình", "kiến trúc", "xây dựng",
                 "bảo tàng", "lịch sử", "văn hóa"],
    "food": ["món ăn", "món ", "ẩm thực", "đặc sản", "thức uống", "nguyên liệu"],
    "festival": ["lễ hội", "tết"],
}

VIETNAM_MARKERS = ["Việt Nam", *PROVINCES]
# The Vietnam marker must appear early, where the article says where the place is.
LOCATION_CHARS = 400
# Lake titles ("Hồ ...") collide with the common surname Hồ, so they need lake wording.
LAKE_WORDS = ["hồ nước", "hồ chứa", "là một hồ", "hồ tự nhiên", "hồ nhân tạo", "mặt hồ"]
BIOGRAPHY = re.compile(r"\(sinh |sinh ngày|\(\d{4}\s*[–-]\s*\d{4}\)|là một (nhà|ca sĩ|diễn viên|cầu thủ|chính khách)")

LEAD_CHARS = 1500
MIN_TEXT_CHARS = 1500
QUOTAS = {"province": 63, "destination": 160, "heritage": 135, "food": 85, "festival": 57}


def load_candidates() -> tuple[pd.DataFrame, set[str]]:
    """Read only the rows whose title is a candidate (much faster than full text)."""
    shards = sorted(RAW_DIR.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {RAW_DIR}; run scripts/download_wikipedia.py")
    all_titles: set[str] = set()
    parts = []
    for path in shards:
        titles = pq.read_table(path, columns=["title"]).column("title").to_pylist()
        all_titles.update(titles)
        wanted = [t for t in titles if candidate_category(t)[0] is not None]
        parts.append(pq.read_table(path, filters=[("title", "in", wanted)]).to_pandas())
    return pd.concat(parts, ignore_index=True), all_titles


def candidate_category(title: str) -> tuple[str | None, bool]:
    """Return (category, is_curated) for a title, or (None, False)."""
    if title in PROVINCES:
        return "province", True
    for cat, titles in CURATED.items():
        if title in titles:
            return cat, True
    if any(title.startswith(p) for p in EXCLUDED_PREFIXES):
        return None, False
    for cat, prefixes in PREFIXES.items():
        if any(title.startswith(p) for p in prefixes):
            return cat, False
    return None, False


def passes_filters(category: str, curated: bool, title: str, text: str) -> bool:
    if title.startswith("Danh sách") or "(định hướng)" in title or len(text) < MIN_TEXT_CHARS:
        return False
    if curated:
        return True
    lead = text[:LEAD_CHARS]
    lead_lower = lead.lower()
    # "Thái Bình Dương" (Pacific Ocean) would otherwise match the provinces
    # Thái Bình and Bình Dương.
    location = lead[:LOCATION_CHARS].replace("Thái Bình Dương", "")
    if not any(m in location for m in VIETNAM_MARKERS):
        return False
    if BIOGRAPHY.search(lead):
        return False
    if title.startswith("Hồ ") and not any(w in lead_lower for w in LAKE_WORDS):
        return False
    return any(k in lead_lower for k in CATEGORY_KEYWORDS[category])


def select(df: pd.DataFrame) -> pd.DataFrame:
    cats = df["title"].map(candidate_category)
    df = df.assign(category=[c for c, _ in cats], curated=[k for _, k in cats])
    df = df[df["category"].notna()]
    keep = [passes_filters(r.category, r.curated, r.title, r.text) for r in df.itertuples()]
    df = df[keep].assign(n_chars=lambda d: d["text"].str.len())

    picked = []
    for cat, quota in QUOTAS.items():
        part = df[df["category"] == cat].sort_values(["curated", "n_chars"], ascending=False)
        picked.append(part.head(quota))
    return pd.concat(picked, ignore_index=True)


def main() -> None:
    df, all_titles = load_candidates()
    print(f"{len(all_titles):,} titles in dump, {len(df):,} candidates")
    selected = select(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with ARTICLES_PATH.open("w", encoding="utf-8") as f:
        for r in selected.itertuples():
            row = {"id": r.id, "title": r.title, "url": r.url, "category": r.category,
                   "curated": bool(r.curated), "text": r.text}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"selected {len(selected)} articles -> {ARTICLES_PATH}")
    print(dict(Counter(selected["category"])))
    missing = [t for ts in CURATED.values() for t in ts if t not in all_titles]
    print(f"curated titles not found in dump ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
