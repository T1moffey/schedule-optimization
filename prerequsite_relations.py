from __future__ import annotations

from typing import Any
import re

import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    # Нормализует вектор или строки матрицы по L2-норме.
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        s = np.linalg.norm(x)
        return x / s if s > 0 else x
    s = np.linalg.norm(x, axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    return x / s


def _clean_text(text: str) -> str:
    # Приводит текст к упрощенному виду для строковых сравнений.
    text = text.strip().lower().replace("ё", "е").replace("_", " ")
    return re.sub(r"\s+", " ", text)


def _clean_topic_name(name: str) -> str:
    # Очищает название темы и убирает возможный числовой префикс.
    return re.sub(r"^\d+[_\-\s]*", "", _clean_text(name)).strip()


def _segment_vector(x: np.ndarray, aggregation: str = "mean") -> np.ndarray:
    # Сводит эмбеддинги одного сегмента к одному итоговому вектору.
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return _norm(x)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Сегмент должен быть вектором или матрицей shape=(n_sentences, dim).")
    if aggregation == "median":
        return _norm(np.median(x, axis=0))
    return _norm(x.mean(axis=0))


def _local_vectors(
    segment_embeddings: list[np.ndarray],
    *,
    local_unit: str = "segment",
    aggregation: str = "mean",
) -> np.ndarray:
    # Собирает локальные векторы темы либо по сегментам, либо по отдельным предложениям.
    vectors: list[np.ndarray] = []

    for segment_embs in segment_embeddings:
        x = np.asarray(segment_embs, dtype=float)

        if local_unit == "sentence":
            if x.ndim == 1:
                vectors.append(_norm(x))
            elif x.ndim == 2 and x.shape[0] > 0:
                vectors.extend(_norm(x))
            else:
                raise ValueError("Ожидается вектор или матрица sentence embeddings.")
        else:
            vectors.append(_segment_vector(x, aggregation=aggregation))

    if not vectors:
        raise ValueError("Не удалось собрать локальные векторы темы.")

    return np.vstack(vectors)


def _topic_prototype(
    segment_embeddings: list[np.ndarray],
    *,
    prototype_mode: str = "all_sentences_mean",
    aggregation: str = "mean",
) -> np.ndarray:
    # Строит один опорный вектор темы по всем ее сегментам.
    if prototype_mode == "segment_means_mean":
        return _norm(_local_vectors(segment_embeddings, local_unit="segment", aggregation=aggregation).mean(axis=0))

    all_sentences: list[np.ndarray] = []
    for segment_embs in segment_embeddings:
        x = np.asarray(segment_embs, dtype=float)
        if x.ndim == 1:
            all_sentences.append(x.reshape(1, -1))
        elif x.ndim == 2 and x.shape[0] > 0:
            all_sentences.append(x)
        else:
            raise ValueError("Ожидается вектор или матрица sentence embeddings.")

    if not all_sentences:
        raise ValueError("Не удалось собрать предложения темы.")

    return _norm(np.vstack(all_sentences).mean(axis=0))


def _csr(
    source_segment_embeddings: list[np.ndarray],
    target_topic_prototype: np.ndarray,
    *,
    local_unit: str = "segment",
    aggregation: str = "mean",
) -> dict[str, float]:
    # Считает направленную семантическую близость локальных фрагментов темы к прототипу другой темы.
    scores = _local_vectors(
        source_segment_embeddings,
        local_unit=local_unit,
        aggregation=aggregation,
    ) @ _norm(target_topic_prototype)

    return {
        "raw_sum": float(scores.sum()),
        "mean": float(scores.mean()),
        "max": float(scores.max()),
        "n_units": int(len(scores)),
    }


def _cer_like(source_segment_texts: list[str], target_topic_name: str) -> dict[str, float | int]:
    # Считает простую текстовую поддержку связи через вхождение названия темы в тексты сегментов.
    if not source_segment_texts:
        return {"count": 0, "rate": 0.0}

    target = _clean_topic_name(target_topic_name)
    if not target:
        return {"count": 0, "rate": 0.0}

    hits = sum(int(target in _clean_text(text)) for text in source_segment_texts)
    return {
        "count": int(hits),
        "rate": float(hits / len(source_segment_texts)),
    }


def build_topic_payloads(
    topic_results: dict[str, Any],
    segmented_embeddings: dict[str, list[np.ndarray]],
    *,
    ignore_outliers: bool = True,
) -> dict[int, dict[str, Any]]:
    # Собирает по каждой теме все ее тексты, эмбеддинги сегментов и источники происхождения.
    topic_info = topic_results["topic_info"]
    assignments = topic_results["segment_topic_assignments"]
    payloads: dict[int, dict[str, Any]] = {}

    for lesson_name, segments in assignments.items():
        lesson_segment_embeddings = segmented_embeddings[lesson_name]

        for seg in segments:
            topic_id = int(seg["topic_id"])
            if ignore_outliers and topic_id == -1:
                continue

            segment_id = int(seg["segment_id"])
            payload = payloads.setdefault(
                topic_id,
                {
                    "topic_id": topic_id,
                    "topic_name": str(topic_info.get(topic_id, {}).get("topic_name", f"topic_{topic_id}")),
                    "segment_texts": [],
                    "segment_embeddings": [],
                    "segment_sources": [],
                },
            )

            payload["segment_texts"].append(str(seg["text"]))
            payload["segment_embeddings"].append(np.asarray(lesson_segment_embeddings[segment_id], dtype=float))
            payload["segment_sources"].append(
                {
                    "lesson_name": lesson_name,
                    "segment_id": segment_id,
                    "n_sentences": int(seg.get("n_sentences", 0)),
                    "n_tokens": int(seg.get("n_tokens", 0)),
                }
            )

    return payloads


def _infer_from_payloads(
    topic_a_payload: dict[str, Any],
    topic_b_payload: dict[str, Any],
    *,
    local_unit: str = "segment",
    aggregation: str = "mean",
    prototype_mode: str = "all_sentences_mean",
    min_prs_mean: float = 0.35,
    min_direction_margin: float = 0.06,
    cer_weight: float = 0.10,
) -> dict[str, Any]:
    # Определяет, есть ли между двумя темами prerequisite-связь и в каком направлении.
    topic_a_name = str(topic_a_payload["topic_name"])
    topic_b_name = str(topic_b_payload["topic_name"])

    topic_a_texts = list(topic_a_payload["segment_texts"])
    topic_b_texts = list(topic_b_payload["segment_texts"])
    topic_a_embs = list(topic_a_payload["segment_embeddings"])
    topic_b_embs = list(topic_b_payload["segment_embeddings"])

    proto_a = _topic_prototype(topic_a_embs, prototype_mode=prototype_mode, aggregation=aggregation)
    proto_b = _topic_prototype(topic_b_embs, prototype_mode=prototype_mode, aggregation=aggregation)

    csr_a_b = _csr(topic_a_embs, proto_b, local_unit=local_unit, aggregation=aggregation)
    csr_b_a = _csr(topic_b_embs, proto_a, local_unit=local_unit, aggregation=aggregation)

    cer_a_b = _cer_like(topic_a_texts, topic_b_name)
    cer_b_a = _cer_like(topic_b_texts, topic_a_name)

    evidence_a_to_b = float(csr_b_a["mean"] + cer_weight * cer_b_a["rate"])
    evidence_b_to_a = float(csr_a_b["mean"] + cer_weight * cer_a_b["rate"])
    prs_mean = float(max(csr_a_b["mean"], csr_b_a["mean"]))
    prs_raw = float(max(csr_a_b["raw_sum"], csr_b_a["raw_sum"]))
    direction_margin = float(abs(evidence_a_to_b - evidence_b_to_a))

    if prs_mean < min_prs_mean:
        direction = "none"
        prerequisite_exists = False
        prerequisite_topic = None
        dependent_topic = None
        reason = "Связь слишком слабая: максимальный усредненный CSR ниже порога."
    elif direction_margin < min_direction_margin:
        direction = "undetermined"
        prerequisite_exists = True
        prerequisite_topic = None
        dependent_topic = None
        reason = "Пара выглядит кандидатной, но два направления слишком близки."
    elif evidence_a_to_b > evidence_b_to_a:
        direction = "a->b"
        prerequisite_exists = True
        prerequisite_topic = topic_a_name
        dependent_topic = topic_b_name
        reason = "Тема A сильнее проявляется в локальных фрагментах темы B."
    else:
        direction = "b->a"
        prerequisite_exists = True
        prerequisite_topic = topic_b_name
        dependent_topic = topic_a_name
        reason = "Тема B сильнее проявляется в локальных фрагментах темы A."

    return {
        "topic_a_name": topic_a_name,
        "topic_b_name": topic_b_name,
        "csr": {
            "csr_a_b": csr_a_b,
            "csr_b_a": csr_b_a,
            "prs_raw": prs_raw,
            "prs_mean": prs_mean,
        },
        "cer_like": {
            "cer_a_b": cer_a_b,
            "cer_b_a": cer_b_a,
        },
        "direction_scores": {
            "evidence_a_to_b": evidence_a_to_b,
            "evidence_b_to_a": evidence_b_to_a,
            "direction_margin": direction_margin,
        },
        "decision": {
            "prerequisite_exists": prerequisite_exists,
            "direction": direction,
            "prerequisite_topic": prerequisite_topic,
            "dependent_topic": dependent_topic,
            "reason": reason,
        },
    }


def infer_prerequisite_from_topic_ids(
    topic_a_id: int,
    topic_b_id: int,
    topic_payloads: dict[int, dict[str, Any]],
    *,
    local_unit: str = "segment",
    aggregation: str = "mean",
    prototype_mode: str = "all_sentences_mean",
    min_prs_mean: float = 0.35,
    min_direction_margin: float = 0.06,
    cer_weight: float = 0.10,
) -> dict[str, Any]:
    # Запускает определение prerequisite-связи для пары тем по их идентификаторам.
    if topic_a_id not in topic_payloads:
        raise KeyError(f"Тема {topic_a_id} отсутствует в topic_payloads.")
    if topic_b_id not in topic_payloads:
        raise KeyError(f"Тема {topic_b_id} отсутствует в topic_payloads.")

    result = _infer_from_payloads(
        topic_payloads[topic_a_id],
        topic_payloads[topic_b_id],
        local_unit=local_unit,
        aggregation=aggregation,
        prototype_mode=prototype_mode,
        min_prs_mean=min_prs_mean,
        min_direction_margin=min_direction_margin,
        cer_weight=cer_weight,
    )
    result["topic_a_id"] = int(topic_a_id)
    result["topic_b_id"] = int(topic_b_id)
    return result


def collect_topic_prerequisite_results(
    topic_payloads: dict[int, dict[str, Any]],
    *,
    local_unit: str = "segment",
    aggregation: str = "mean",
    prototype_mode: str = "all_sentences_mean",
    min_prs_mean: float = 0.35,
    min_direction_margin: float = 0.06,
    cer_weight: float = 0.10,
) -> list[dict[str, Any]]:
    # Перебирает все пары тем и собирает результаты prerequisite-анализа для каждой пары.
    results: list[dict[str, Any]] = []
    topic_ids = sorted(topic_payloads)

    for i, topic_a_id in enumerate(topic_ids):
        for topic_b_id in topic_ids[i + 1:]:
            results.append(
                infer_prerequisite_from_topic_ids(
                    topic_a_id=topic_a_id,
                    topic_b_id=topic_b_id,
                    topic_payloads=topic_payloads,
                    local_unit=local_unit,
                    aggregation=aggregation,
                    prototype_mode=prototype_mode,
                    min_prs_mean=min_prs_mean,
                    min_direction_margin=min_direction_margin,
                    cer_weight=cer_weight,
                )
            )

    return results