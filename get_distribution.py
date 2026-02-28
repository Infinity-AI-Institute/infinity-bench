from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone

from data_loader import get_infinity_dataset_names, load_classification_datasets

EXCLUDED_DATASETS = {"KDDCup99"}


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out


def _clone_model(model: Any) -> Any:
    try:
        return clone(model)
    except Exception:
        return deepcopy(model)


def _predict_distribution(model: Any, X_test: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X_test), dtype=np.float64)

        if probs.ndim == 1:
            probs = np.column_stack([1.0 - probs, probs])

        class_ids = np.asarray(getattr(model, "classes_", np.arange(probs.shape[1])))
        aligned = np.zeros((probs.shape[0], n_classes), dtype=np.float64)
        for src_col, class_id in enumerate(class_ids):
            class_idx = int(class_id)
            if 0 <= class_idx < n_classes:
                aligned[:, class_idx] = probs[:, src_col]

        row_sums = aligned.sum(axis=1, keepdims=True)
        nonzero = row_sums.squeeze() > 0
        aligned[nonzero] = aligned[nonzero] / row_sums[nonzero]
        return aligned.astype(np.float32)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_test), dtype=np.float64)
        if scores.ndim == 1:
            p1 = _stable_sigmoid(scores)
            probs = np.column_stack([1.0 - p1, p1])
        else:
            probs = _stable_softmax(scores)
        return probs.astype(np.float32)

    # Fallback when only hard labels are available.
    labels = np.asarray(model.predict(X_test), dtype=int)
    probs = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
    probs[np.arange(labels.shape[0]), labels] = 1.0
    return probs


def get_distribution_tensor(
    model: Any,
    dataset_names: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    logging: bool = False,
    freeze_splits: bool = True,
    freeze_dir: str = ".infinity_frozen_splits",
    refresh_frozen_splits: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Train the model on each dataset and return pre-argmax prediction distributions.

    Returns:
        dataset_order:
            Dataset names in output order.
        distribution_tensor:
            Object-dtype numpy array of shape (num_datasets,). Each element is a
            float32 array of shape (num_test_points, num_classes).
    """
    if dataset_names is None:
        dataset_names = get_infinity_dataset_names()
    dataset_names = [name for name in dataset_names if name not in EXCLUDED_DATASETS]
    if len(dataset_names) == 0:
        raise ValueError("dataset_names must be non-empty.")

    datasets = load_classification_datasets(
        dataset_names=dataset_names,
        test_size=test_size,
        random_state=random_state,
        logging=logging,
        freeze_splits=freeze_splits,
        freeze_dir=freeze_dir,
        refresh_frozen_splits=refresh_frozen_splits,
    )

    distribution_tensor = np.empty((len(dataset_names),), dtype=object)

    for i, dataset_name in enumerate(dataset_names):
        X_train, X_test, y_train, _ = datasets[dataset_name]
        model_i = _clone_model(model)
        model_i.fit(np.asarray(X_train), y_train)

        n_classes = int(np.unique(y_train).shape[0])
        distribution_tensor[i] = _predict_distribution(
            model=model_i,
            X_test=np.asarray(X_test),
            n_classes=n_classes,
        )

        if logging:
            shape = distribution_tensor[i].shape
            print(f"{dataset_name:24s} distribution_shape={shape}")

    return dataset_names, distribution_tensor


def write_distribution_json(
    model: Any,
    output_path: str = "distribution_predictions.json",
    dataset_names: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    logging: bool = False,
    freeze_splits: bool = True,
    freeze_dir: str = ".infinity_frozen_splits",
    refresh_frozen_splits: bool = False,
) -> Dict[str, Any]:
    """
    Write pre-argmax prediction distributions as JSON rows.

    Each row contains:
      - dataset
      - predicted_class (argmax)
      - real_class
      - class_0_pred ... class_{K-1}_pred, where K is max class count across datasets.
        For datasets with fewer classes, extra class columns are null.
    """
    if dataset_names is None:
        dataset_names = get_infinity_dataset_names()
    dataset_names = [name for name in dataset_names if name not in EXCLUDED_DATASETS]
    if len(dataset_names) == 0:
        raise ValueError("dataset_names must be non-empty.")

    datasets = load_classification_datasets(
        dataset_names=dataset_names,
        test_size=test_size,
        random_state=random_state,
        logging=logging,
        freeze_splits=freeze_splits,
        freeze_dir=freeze_dir,
        refresh_frozen_splits=refresh_frozen_splits,
    )

    max_classes = max(int(np.unique(y_train).shape[0]) for _, _, y_train, _ in datasets.values())

    rows: List[Dict[str, Any]] = []
    for dataset_name in dataset_names:
        X_train, X_test, y_train, y_test = datasets[dataset_name]
        n_classes = int(np.unique(y_train).shape[0])

        model_i = _clone_model(model)
        model_i.fit(np.asarray(X_train), y_train)
        dist = _predict_distribution(model=model_i, X_test=np.asarray(X_test), n_classes=n_classes)

        y_true = np.asarray(y_test, dtype=int)
        y_pred = np.argmax(dist, axis=1).astype(int)

        for i in range(dist.shape[0]):
            row: Dict[str, Any] = {
                "dataset": dataset_name,
                "predicted_class": int(y_pred[i]),
                "real_class": int(y_true[i]),
            }
            for class_idx in range(max_classes):
                key = f"class_{class_idx}_pred"
                if class_idx < dist.shape[1]:
                    row[key] = float(dist[i, class_idx])
                else:
                    row[key] = None
            rows.append(row)

        if logging:
            print(f"{dataset_name:24s} rows={dist.shape[0]} classes={n_classes}")

    payload: Dict[str, Any] = {
        "max_classes": max_classes,
        "datasets": dataset_names,
        "rows": rows,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if logging:
        print(f"Wrote JSON: {output_path} (rows={len(rows)}, max_classes={max_classes})")

    return payload


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000)
    write_distribution_json(model=model, logging=True)
