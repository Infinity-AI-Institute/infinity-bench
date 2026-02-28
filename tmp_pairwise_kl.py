from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _extract_distribution(row: Dict[str, Any], target_dim: int) -> np.ndarray:
    dist = np.zeros((target_dim,), dtype=np.float64)
    for key, value in row.items():
        if key.startswith("class_") and key.endswith("_pred"):
            class_idx = int(key.split("_")[1])
            if 0 <= class_idx < target_dim and value is not None:
                dist[class_idx] = float(value)
    s = float(np.sum(dist))
    if s > 0:
        dist = dist / s
    return dist


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _infer_global_class_dim(model_rows: Dict[str, List[Dict[str, Any]]]) -> int:
    max_idx = -1
    for rows in model_rows.values():
        for row in rows:
            for key in row.keys():
                if key.startswith("class_") and key.endswith("_pred"):
                    max_idx = max(max_idx, int(key.split("_")[1]))
    if max_idx < 0:
        raise ValueError("No class_*_pred keys found.")
    return max_idx + 1


def _validate_alignment(model_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    model_names = list(model_rows.keys())
    n_rows = len(model_rows[model_names[0]])
    base = model_rows[model_names[0]]
    for name in model_names:
        if len(model_rows[name]) != n_rows:
            raise ValueError("Row count mismatch across models.")
    for name in model_names[1:]:
        rows = model_rows[name]
        for i in range(n_rows):
            if (
                base[i].get("dataset") != rows[i].get("dataset")
                or int(base[i].get("real_class")) != int(rows[i].get("real_class"))
            ):
                raise ValueError(f"Row alignment mismatch: {model_names[0]} vs {name} at row {i}")


def _load_model_rows(input_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    root = Path(input_dir)
    model_files = sorted(
        p
        for p in root.glob("*.json")
        if p.name != "summary.json"
        and "divergence" not in p.name
        and "pairwise_kl_table" not in p.name
    )

    model_rows: Dict[str, List[Dict[str, Any]]] = {}
    for path in model_files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) == 0:
            continue
        model_rows[path.stem] = rows

    if len(model_rows) < 2:
        raise ValueError("Need at least 2 model files to compute pairwise KL.")
    return model_rows


def build_pairwise_kl_table(
    input_dir: str = "sklearn_distribution_outputs",
    output_csv: str = "sklearn_distribution_outputs/pairwise_kl_table.csv",
) -> Tuple[List[str], np.ndarray]:
    model_rows = _load_model_rows(input_dir=input_dir)
    _validate_alignment(model_rows)
    global_dim = _infer_global_class_dim(model_rows)

    model_names = list(model_rows.keys())
    n_models = len(model_names)
    n_rows = len(model_rows[model_names[0]])

    dist_tensor = np.stack(
        [
            np.stack([_extract_distribution(r, target_dim=global_dim) for r in model_rows[name]], axis=0)
            for name in model_names
        ],
        axis=0,
    )  # [m, r, c]

    kl_matrix = np.zeros((n_models, n_models), dtype=np.float64)
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                kl_matrix[i, j] = 0.0
            else:
                vals = [_kl_divergence(dist_tensor[i, k], dist_tensor[j, k]) for k in range(n_rows)]
                kl_matrix[i, j] = float(np.mean(vals))

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + model_names)
        for i, name in enumerate(model_names):
            writer.writerow([name] + [f"{kl_matrix[i, j]:.8f}" for j in range(n_models)])

    return model_names, kl_matrix


if __name__ == "__main__":
    names, matrix = build_pairwise_kl_table()
    print(f"Wrote pairwise KL table for {len(names)} models.")
    print("Rows = P, Columns = Q, entries = mean KL(P||Q)")
    print("Output: sklearn_distribution_outputs/pairwise_kl_table.csv")
