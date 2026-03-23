import math
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def make_ngrams(text: str, n: int = 10, step: int = 10) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    if len(words) <= n:
        return [" ".join(words)]
    return [" ".join(words[i:i + n]) for i in range(0, len(words) - n + 1, step)]


def csr(
    description_text: str,
    concept_title: str,
    model: SentenceTransformer = DEFAULT_MODEL,
    n: int = 10,
    step: int = 10,
    concept_emb: np.ndarray | None = None,
    chunk_embs: np.ndarray | None = None,
) -> float:
    """
    CSR(description_text, concept_title) =
    сумма cosine similarity между всеми n-граммами описания и названием концепта.
    """
    chunks = make_ngrams(description_text, n=n, step=step)

    if chunk_embs is None:
        chunk_embs = np.asarray(model.encode(chunks))
    if concept_emb is None:
        concept_emb = np.asarray(model.encode([concept_title]))[0]

    chunk_embs = normalize_rows(np.asarray(chunk_embs))
    concept_emb = concept_emb / max(np.linalg.norm(concept_emb), 1e-12)

    sims = chunk_embs @ concept_emb
    return float(sims.sum())


def cer(
    description_text: str,
    concept_title: str,
    n: int = 10,
    step: int = 10,
    case_sensitive: bool = False,
) -> int:
    """
    CER(description_text, concept_title) =
    количество n-грамм описания, в которых есть точное текстовое вхождение названия концепта.
    """
    chunks = make_ngrams(description_text, n=n, step=step)

    if not case_sensitive:
        concept_title = concept_title.lower()
        chunks = [chunk.lower() for chunk in chunks]

    return sum(concept_title in chunk for chunk in chunks)


def transitive_reduce(g: nx.DiGraph) -> nx.DiGraph:
    if not nx.is_directed_acyclic_graph(g):
        return g
    reduced = nx.transitive_reduction(g)
    out = nx.DiGraph()
    out.add_nodes_from(g.nodes())
    out.add_edges_from(reduced.edges())
    return out


def ask_expert(topic_a: str, topic_b: str, score_ab: float, score_ba: float, prs: float, mode: str) -> int:
    print("\n" + "=" * 80)
    print(f"Пара: {topic_a!r}  <->  {topic_b!r}")
    print(f"{mode.upper()}({topic_a} -> {topic_b}) = {score_ab:.6f}")
    print(f"{mode.upper()}({topic_b} -> {topic_a}) = {score_ba:.6f}")
    print(f"PRS = {prs:.6f}")
    print("Введите решение:")
    print(f"  0 -> {topic_a} prerequisite для {topic_b}   ({topic_a} -> {topic_b})")
    print(f"  1 -> {topic_b} prerequisite для {topic_a}   ({topic_b} -> {topic_a})")
    print("  2 -> остановить разметку")

    while True:
        ans = input("Ваш выбор [0/1/2]: ").strip()
        if ans in {"0", "1", "2"}:
            return int(ans)


def prepare_embeddings(
    topic_to_text: dict[str, str],
    model: SentenceTransformer = DEFAULT_MODEL,
    n: int = 10,
    step: int = 10,
):
    """
    Предварительно считаем эмбеддинги, чтобы не вызывать model.encode много раз
    внутри csr(...) для одних и тех же тем.
    """
    topics = list(topic_to_text.keys())

    topic_embs = {
        topic: np.asarray(model.encode([topic]))[0]
        for topic in topics
    }

    chunk_embs = {}
    for topic, text in topic_to_text.items():
        chunks = make_ngrams(text, n=n, step=step)
        chunk_embs[topic] = np.asarray(model.encode(chunks))

    return topic_embs, chunk_embs


def build_ranked_pairs(
    topic_to_text: dict[str, str],
    mode: str = "csr",
    model: SentenceTransformer = DEFAULT_MODEL,
    n: int = 10,
    step: int = 10,
):
    topics = list(topic_to_text.keys())

    topic_embs, topic_chunk_embs = prepare_embeddings(
        topic_to_text=topic_to_text,
        model=model,
        n=n,
        step=step,
    )

    ranked = []
    pair_scores = {}

    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            a, b = topics[i], topics[j]
            text_a, text_b = topic_to_text[a], topic_to_text[b]

            if mode == "csr":
                score_ab = csr(
                    description_text=text_a,
                    concept_title=b,
                    model=model,
                    n=n,
                    step=step,
                    concept_emb=topic_embs[b],
                    chunk_embs=topic_chunk_embs[a],
                )
                score_ba = csr(
                    description_text=text_b,
                    concept_title=a,
                    model=model,
                    n=n,
                    step=step,
                    concept_emb=topic_embs[a],
                    chunk_embs=topic_chunk_embs[b],
                )
            else:
                score_ab = cer(text_a, b, n=n, step=step)
                score_ba = cer(text_b, a, n=n, step=step)

            prs = max(score_ab, score_ba)

            ranked.append((prs, a, b))
            pair_scores[(a, b)] = (score_ab, score_ba)

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked, pair_scores


def run_ace(
    topic_to_text: dict[str, str],
    mode: str = "csr",   # "csr" или "cer"
    model: SentenceTransformer = DEFAULT_MODEL,
    n: int = 10,
    step: int = 10,
    top_t_percent: float = 100.0,
):
    ranked_pairs, pair_scores = build_ranked_pairs(
        topic_to_text=topic_to_text,
        mode=mode,
        model=model,
        n=n,
        step=step,
    )

    g = nx.DiGraph()
    g.add_nodes_from(topic_to_text.keys())

    limit = math.ceil(len(ranked_pairs) * top_t_percent / 100)

    for prs, a, b in ranked_pairs[:limit]:
        if nx.has_path(g, a, b) or nx.has_path(g, b, a):
            continue

        score_ab, score_ba = pair_scores[(a, b)]
        ans = ask_expert(a, b, score_ab, score_ba, prs, mode)

        if ans == 2:
            break
        elif ans == 0:
            g.add_edge(a, b)
        else:
            g.add_edge(b, a)

        g = transitive_reduce(g)

    return g, ranked_pairs, pair_scores


def print_graph(g: nx.DiGraph):
    print("\n" + "=" * 80)
    print("Итоговые ребра графа:")
    for u, v in g.edges():
        print(f"{u} -> {v}")
    print("=" * 80)

# ======================================================================
# ====================== ДАЛЕЕ ТЕСТИРУЮЩАЯ ЧАСТЬ =======================
# ======================================================================

import os
import re
import pandas as pd


def norm_name(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip())


def load_example(
    csv_file="EKG-Dataset-main/0-100.csv",
    descriptions_dir="EKG-Dataset-main/concept_descriptions",
):
    df = pd.read_csv(csv_file)
    df.columns = [c.lower() for c in df.columns]

    concepts = sorted(set(df["concept1"].astype(str)) | set(df["concept2"].astype(str)))

    topic_to_text = {
        c: open(os.path.join(descriptions_dir, f"{norm_name(c)}.txt"), encoding="utf-8").read().strip()
        for c in concepts
    }

    return df, topic_to_text


def test_on_csv(
    csv_file="EKG-Dataset-main\0-100.csv",
    descriptions_dir="EKG-Dataset-main\concept_descriptions",
):
    df, topic_to_text = load_example(csv_file, descriptions_dir)

    ranked_pairs, pair_scores = build_ranked_pairs(
        topic_to_text=topic_to_text,
        mode="csr",
        model=DEFAULT_MODEL,
        n=10,
        step=10,
    )

    calc = {
        tuple(sorted((a, b))): max(score_ab, score_ba)
        for _, a, b in ranked_pairs
        for score_ab, score_ba in [pair_scores[(a, b)]]
    }

    df["calculated_csr_score"] = [
        calc[tuple(sorted((row["concept1"], row["concept2"])))]
        for _, row in df.iterrows()
    ]
    df["abs_diff"] = (df["calculated_csr_score"] - df["csr_score"]).abs()

    print("\nСравнение с датасетом:\n")
    print(df[["concept1", "concept2", "csr_score", "calculated_csr_score", "abs_diff"]].to_string(index=False))

    graph, _, _ = run_ace(
        topic_to_text=topic_to_text,
        mode="csr",
        model=DEFAULT_MODEL,
        n=10,
        step=10,
        top_t_percent=100.0,
    )

    print("\nМинимальный граф:\n")
    for u, v in graph.edges():
        print(f"{u} -> {v}")


if __name__ == "__main__":
    test_on_csv()
