import json
import uuid
import requests
from openai import OpenAI

# =========================
# 1. AUTH KEY
# =========================
AUTH_KEY = "MDE5ZDk0YzUtZDIyYi03NmI4LWE0ZDQtNmY5YzEyYWJhMDkzOjQ2NzkxOTk2LTdjYzUtNDY3My1iZTFlLTJhOWRjZTI0NjlhNQ=="

# =========================
# 2. ФАЙЛЫ
# =========================
INPUT_JSON_PATH = "thesis\\RPD_SPBU\\rpd_2_2.json"
OUTPUT_JSON_PATH = "topics_first_course.json"

# =========================
# 3. ПОЛУЧЕНИЕ ACCESS TOKEN
# =========================
def get_access_token(auth_key: str) -> str:
    token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Bearer {auth_key}",
    }
    data = {
        "scope": "GIGACHAT_API_PERS"
    }

    response = requests.post(token_url, headers=headers, data=data, timeout=30)
    response.raise_for_status()
    return response.json()["access_token"]


# =========================
# 4. ЗАГРУЗКА ПЕРВОГО КУРСА ИЗ JSON
# =========================
def load_first_course(path: str) -> tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    first_course_title = next(iter(data.keys()))
    first_course_text = data[first_course_title]

    return first_course_title, first_course_text


# =========================
# 5. ПОДГОТОВКА НЕПУСТЫХ СТРОК
# =========================
def get_nonempty_lines(source_text: str) -> list[str]:
    return [line.strip() for line in source_text.splitlines() if line.strip()]


# =========================
# 6. ПРОМПТЫ
# =========================
SYSTEM_PROMPT = """
Ты извлекаешь из строк текста понятия, относящиеся к теме курса.

Правила:
- анализируй каждую непустую строку отдельно;
- извлекай понятия, которые явно присутствуют в строке или однозначно читаются с учетом артефактов извлечения текста;
- учитывай название курса как фильтр релевантности;
- не добавляй ничего от себя;
- если в строке нет подходящих понятий, возвращай пустой список;
- верни только JSON.
""".strip()


# =========================
# 7. ЗАПРОС К МОДЕЛИ
# =========================
def extract_concepts_for_course(auth_key: str, system_prompt: str, user_prompt: str) -> dict:
    access_token = get_access_token(auth_key)

    client = OpenAI(
        api_key=access_token,
        base_url="https://gigachat.devices.sberbank.ru/api/v1"
    )

    response = client.chat.completions.create(
        model="GigaChat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    raw_answer = response.choices[0].message.content
    parsed = json.loads(raw_answer)
    return parsed


# =========================
# 8. ОБЪЕДИНЕНИЕ ТЕМ
# =========================
def collect_concepts(parsed_response: dict) -> list[str]:
    result = []
    seen = set()

    for item in parsed_response.get("lines", []):
        concepts = item.get("concepts", [])

        if not isinstance(concepts, list):
            continue

        for concept in concepts:
            if not isinstance(concept, str):
                continue

            concept = concept.strip()
            if not concept:
                continue

            if concept not in seen:
                seen.add(concept)
                result.append(concept)

    return result


# =========================
# 9. СОХРАНЕНИЕ
# =========================
def save_result(output_path: str, course_title: str, concepts: list[str]) -> None:
    result = {
        course_title: concepts
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# =========================
# 10. ОСНОВНОЙ ЗАПУСК
# =========================
def main():
    course_title, source_text = load_first_course(INPUT_JSON_PATH)
    lines = get_nonempty_lines(source_text)

    input_lines = [
        {"line_number": i, "text": line}
        for i, line in enumerate(lines, start=1)
    ]

    USER_PROMPT = f"""
Название курса:
{course_title}

Ниже дан список непустых строк текста.

Для каждой строки выдели понятия, которые:
1) явно присутствуют в тексте строки;
2) соответствуют теме курса.

Важно:
- в тексте могут быть артефакты извлечения: склеенные слова без пробелов, лишние переносы, сломанные пробелы;
- если понятие читается однозначно с учетом таких артефактов, его можно извлекать;
- не пропускай строки при анализе;
- не добавляй ничего от себя.

Верни строго JSON в формате:
{{
  "lines": [
    {{
      "line_number": 1,
      "concepts": ["понятие 1", "понятие 2"]
    }},
    {{
      "line_number": 2,
      "concepts": []
    }}
  ]
}}

Строки:
{json.dumps(input_lines, ensure_ascii=False, indent=2)}
""".strip()

    parsed = extract_concepts_for_course(
        auth_key=AUTH_KEY,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT
    )

    concepts = collect_concepts(parsed)
    save_result(OUTPUT_JSON_PATH, course_title, concepts)


if __name__ == "__main__":
    main()