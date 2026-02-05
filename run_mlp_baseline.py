#!/usr/bin/env python3
"""Quick script to benchmark sklearn MLPClassifier on the 20 datasets."""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from data_loader import load_classification_datasets


DEFAULT_DATASETS: List[str] = [
    "Iris",
    "Wine",
    "Breast Cancer",
    "Digits",
    "Balance Scale",
    "Blood Transfusion",
    "Haberman",
    "Seeds",
    "Teaching Assistant",
    "Zoo",
    "Planning Relax",
    "Ionosphere",
    "Sonar",
    "Glass",
    "Vehicle",
    "Liver Disorders",
    "Heart Statlog",
    "Pima Indians Diabetes",
    "Australian",
    "Monks-1",
]


def run_mlp(
    dataset_names: List[str],
    random_state: int,
    max_iter: int,
) -> Dict[str, float]:
    datasets = load_classification_datasets(dataset_names, logging=True)

    scores: Dict[str, float] = {}
    for dataset_name, (X_train, X_test, y_train, y_test) in datasets.items():
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            random_state=random_state,
            max_iter=max_iter,
        )
        t0 = time.perf_counter()
        model.fit(np.asarray(X_train), y_train)
        y_pred = model.predict(np.asarray(X_test))
        score = accuracy_score(y_test, y_pred)
        dt = time.perf_counter() - t0
        scores[dataset_name] = float(score)
        print(f"{dataset_name:24s}  acc={score:.4f}  total_s={dt:.2f}")

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sklearn MLPClassifier baseline.")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON scores.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for MLPClassifier.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Max training iterations for MLPClassifier.",
    )
    args = parser.parse_args()

    scores = run_mlp(
        dataset_names=DEFAULT_DATASETS,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )

    print("\nJSON:")
    print(json.dumps(scores, indent=2, sort_keys=True))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
