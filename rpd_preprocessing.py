from pathlib import Path
import json
import re

import pdfplumber


ROOT_DIR = Path(r"RPD_SPBU")
OUTPUT_FILE = ROOT_DIR / "rpd_2_2.json"


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=[A-Za-zА-Яа-яЁё0-9])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_course_title(pdf_path: Path) -> str:
    name = pdf_path.stem
    name = re.sub(r"^\d+[_\-\s]*", "", name)
    name = name.replace("_", " ")
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name


def extract_pdf_text(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return normalize_text("\n".join(parts))


def extract_section_22(full_text: str) -> str | None:
    start_match = re.search(
        r"2\.2\.?[\s\S]{0,120}?Структура\s*и\s*содержание\s*учебных\s*занятий",
        full_text,
        re.IGNORECASE,
    )
    if start_match is None:
        return None

    end_match = re.search(
        r"\bРаздел\s*3\b",
        full_text[start_match.end():],
        re.IGNORECASE,
    )
    if end_match is None:
        return None

    end = start_match.end() + end_match.start()
    section_text = full_text[start_match.end():end]

    return normalize_text(section_text)


def main():
    result = {}

    for pdf_path in sorted(ROOT_DIR.rglob("*.pdf")):
        course_title = extract_course_title(pdf_path)
        full_text = extract_pdf_text(pdf_path)
        section_22 = extract_section_22(full_text)

        if section_22 is None:
            print(f"[NOT FOUND] В файле '{pdf_path.name}' не найден раздел 2.2")
            continue

        result[course_title] = section_22
        print(f"[OK] {course_title}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nГотово: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()