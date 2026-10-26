"""Reproducible masked-edge reconstruction on the Elliptic transaction graph.

The experiment is intentionally distinct from predicting future Bitcoin flows:
Elliptic has no edges between its 49 time steps. Each time step is treated as
an independent graph, and the classifier is tested on later time steps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ["common_neighbors", "jaccard", "adamic_adar", "degree_product"]


def load_graphs(data_dir: Path) -> dict[int, nx.Graph]:
    """Read only node IDs/time steps and edges; the 165 node features are unused."""
    features_path = data_dir / "elliptic_txs_features.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"
    if not features_path.is_file() or not edges_path.is_file():
        raise FileNotFoundError(
            "Put elliptic_txs_features.csv and elliptic_txs_edgelist.csv in "
            f"{data_dir}. See README.md for the original data source."
        )

    nodes = pd.read_csv(
        features_path,
        header=None,
        usecols=[0, 1],
        names=["tx_id", "time_step"],
        dtype={"tx_id": "int64", "time_step": "int16"},
    )
    edges = pd.read_csv(
        edges_path,
        usecols=["txId1", "txId2"],
        dtype={"txId1": "int64", "txId2": "int64"},
    )
    step_by_id = dict(zip(nodes.tx_id, nodes.time_step, strict=True))
    graphs = {int(step): nx.Graph() for step in sorted(nodes.time_step.unique())}
    for step, group in nodes.groupby("time_step", sort=True):
        graphs[int(step)].add_nodes_from(group.tx_id.astype(int))

    cross_step = 0
    unknown = 0
    for u, v in edges.itertuples(index=False, name=None):
        su = step_by_id.get(u)
        sv = step_by_id.get(v)
        if su is None or sv is None:
            unknown += 1
        elif su != sv:
            cross_step += 1
        elif u != v:
            graphs[int(su)].add_edge(int(u), int(v))
    if unknown or cross_step:
        raise ValueError(
            f"Unexpected edge endpoints: {unknown} unknown IDs, "
            f"{cross_step} cross-step edges."
        )
    return graphs


def sample_masked_edges(
    full_graph: nx.Graph, *, seed: int, holdout: float = 0.2
) -> tuple[nx.Graph, list[tuple[int, int]], list[tuple[int, int]]]:
    """Hide edges and sample an equal number of true nonedges.

    Negative pairs are checked against the *complete* graph, so a hidden
    positive edge cannot accidentally receive a negative label.
    """
    if not 0 < holdout < 1:
        raise ValueError("holdout must be between 0 and 1")
    edges = list(full_graph.edges())
    if len(edges) < 5:
        return full_graph.copy(), [], []
    rng = np.random.default_rng(seed)
    n_hidden = max(1, int(round(len(edges) * holdout)))
    chosen = rng.choice(len(edges), size=n_hidden, replace=False)
    positives = [edges[int(i)] for i in chosen]
    context = full_graph.copy()
    context.remove_edges_from(positives)

    nodes = list(full_graph.nodes())
    negatives: list[tuple[int, int]] = []
    sampled: set[tuple[int, int]] = set()
    max_nonedges = len(nodes) * (len(nodes) - 1) // 2 - full_graph.number_of_edges()
    if max_nonedges < n_hidden:
        raise ValueError("Not enough nonedges for balanced negative sampling")
    while len(negatives) < n_hidden:
        i, j = rng.integers(len(nodes), size=2)
        if i == j:
            continue
        u, v = sorted((nodes[int(i)], nodes[int(j)]))
        pair = (u, v)
        if full_graph.has_edge(u, v) or pair in sampled:
            continue
        sampled.add(pair)
        negatives.append(pair)
    return context, positives, negatives


def pair_features(graph: nx.Graph, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Compute topology features using only the observed context graph."""
    neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    values = np.empty((len(pairs), len(FEATURE_NAMES)), dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        nu, nv = neighbors[u], neighbors[v]
        common = nu & nv
        union_size = len(nu | nv)
        adamic = sum(1.0 / math.log(len(neighbors[w])) for w in common if len(neighbors[w]) > 1)
        values[i] = (
            len(common),
            len(common) / union_size if union_size else 0.0,
            adamic,
            len(nu) * len(nv),
        )
    return values


def build_dataset(
    graphs: dict[int, nx.Graph], *, seed: int, holdout: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    skipped: list[int] = []
    for step, graph in graphs.items():
        context, positives, negatives = sample_masked_edges(
            graph, seed=seed + step, holdout=holdout
        )
        if not positives:
            skipped.append(step)
            continue
        pairs = positives + negatives
        features.append(pair_features(context, pairs))
        labels.append(np.r_[np.ones(len(positives)), np.zeros(len(negatives))].astype(int))
        steps.append(np.full(len(pairs), step, dtype=int))
    if not features:
        raise ValueError("No usable graph snapshots")
    return (
        np.vstack(features),
        np.concatenate(labels),
        np.concatenate(steps),
        {"snapshots_used": len(features), "snapshots_skipped": len(skipped)},
    )


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    scores = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "average_precision": round(float(average_precision_score(y, scores)), 4),
    }


def run(data_dir: Path, *, seed: int = 42, holdout: float = 0.2) -> dict:
    graphs = load_graphs(data_dir)
    X, y, step, counts = build_dataset(graphs, seed=seed, holdout=holdout)
    masks = {
        "train": step <= 34,
        "validation": (step >= 35) & (step <= 40),
        "test": step >= 41,
    }
    for name, mask in masks.items():
        if mask.sum() == 0 or len(np.unique(y[mask])) != 2:
            raise ValueError(f"Split {name} is empty or has only one class")

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=-1
        ),
    }
    validation = {}
    for name, model in candidates.items():
        model.fit(X[masks["train"]], y[masks["train"]])
        validation[name] = evaluate(model, X[masks["validation"]], y[masks["validation"]])
    selected = max(validation, key=lambda name: validation[name]["average_precision"])
    # Test is touched only after model selection on validation snapshots.
    test = evaluate(candidates[selected], X[masks["test"]], y[masks["test"]])
    return {
        "task": "masked-edge reconstruction within each time step",
        "split": "train 1-34; validation 35-40; test 41-49",
        "seed": seed,
        "holdout_fraction": holdout,
        "features": FEATURE_NAMES,
        "samples": {name: int(mask.sum()) for name, mask in masks.items()},
        **counts,
        "validation": validation,
        "selected_model": selected,
        "test": test,
        "note": "Negative pairs are sampled to match hidden edges; scores are not calibrated to real-world edge prevalence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=0.2)
    args = parser.parse_args()
    result = run(args.data_dir, seed=args.seed, holdout=args.holdout)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
