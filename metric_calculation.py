import pandas as pd
import networkx as nx


def load_graph_from_csv(file_path: str) -> nx.DiGraph:
    df = pd.read_csv(file_path)
    return nx.from_pandas_edgelist(
        df,
        source="source",
        target="target",
        create_using=nx.DiGraph,
    )


def compute_out_closeness_centrality(g: nx.DiGraph) -> dict[str, float]:
    n = len(g)
    scores = {}

    for node in g.nodes():
        lengths = nx.single_source_shortest_path_length(g, node)
        lengths.pop(node, None)

        reachable = len(lengths)
        if reachable == 0:
            scores[node] = 0.0
            continue

        dist_sum = sum(lengths.values())
        scores[node] = (reachable / dist_sum) * (reachable / (n - 1)) if n > 1 else 0.0

    return scores


def compute_learning_effort(g: nx.DiGraph) -> dict[str, int]:
    ancestors_cache = {}

    def get_all_ancestors(node: str) -> set[str]:
        if node in ancestors_cache:
            return ancestors_cache[node]

        preds = list(g.predecessors(node))
        result = set(preds)

        for p in preds:
            result |= get_all_ancestors(p)

        ancestors_cache[node] = result
        return result

    return {
        node: 1 + len(get_all_ancestors(node))
        for node in g.nodes()
    }


def compute_graph_metrics(g: nx.DiGraph) -> pd.DataFrame:
    pagerank = nx.pagerank(g, alpha=0.85)

    betweenness = nx.betweenness_centrality(g, normalized=True)
    out_closeness = compute_out_closeness_centrality(g)
    learning_effort = compute_learning_effort(g)

    rows = []
    for node in g.nodes():
        rows.append({
            "concept": node,
            "pagerank": pagerank[node],
            "betweenness_centrality": betweenness[node],
            "out_closeness_centrality": out_closeness[node],
            "learning_effort": learning_effort[node],
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["pagerank", "learning_effort", "betweenness_centrality"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return df


def save_metrics(
    input_graph_file: str = "ace_graph.csv",
    output_metrics_file: str = "ace_graph_metrics.csv",
):
    g = load_graph_from_csv(input_graph_file)
    metrics_df = compute_graph_metrics(g)
    metrics_df.to_csv(output_metrics_file, index=False, encoding="utf-8-sig")

    print(f"Метрики сохранены в файл: {output_metrics_file}")
    print("\nПервые строки результата:\n")
    print(metrics_df.head(20).to_string(index=False))


if __name__ == "__main__":
    save_metrics(
        input_graph_file="ace_graph.csv",
        output_metrics_file="ace_graph_metrics.csv",
    )