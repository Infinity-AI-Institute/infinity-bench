from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    fetch_covtype,
    fetch_kddcup99,
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


def get_infinity_dataset_names() -> List[str]:
    """Get the fixed Infinity Benchmark dataset names (20 datasets)."""
    return [
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


def get_additional_dataset_names() -> List[str]:
    """Get additional non-Infinity datasets available in this package."""
    return [
        "Adult Census Income",
        "Credit-g",
        "Bank Marketing",
        "Covertype",
        "Electricity",
        "Phoneme",
        "Satimage",
        "Madelon",
        "Amazon Employee Access",
        "KDDCup99",
    ]


def get_all_dataset_names() -> List[str]:
    """Get all available dataset names (Infinity + additional datasets)."""
    return get_infinity_dataset_names() + get_additional_dataset_names()


def get_openml_classification_specs() -> Dict[str, Tuple[int, Optional[int]]]:
    """Map OpenML-backed datasets to (data_id, max_rows)."""
    return {
        # Infinity 20 OpenML datasets
        "Balance Scale": (1463, None),        # 625
        "Blood Transfusion": (1464, None),    # 748
        "Haberman": (43, None),               # 306
        "Seeds": (1499, None),                # 210
        "Teaching Assistant": (48, None),     # 151
        "Zoo": (62, None),                    # 101
        "Planning Relax": (1490, None),       # 182
        "Ionosphere": (59, None),             # 351
        "Sonar": (40, None),                  # 208
        "Glass": (41, None),                  # 214
        "Vehicle": (54, None),                # 846
        "Liver Disorders": (1459, None),      # 345
        "Heart Statlog": (53, None),          # 270
        "Pima Indians Diabetes": (37, None),  # 768
        "Australian": (40945, None),          # 690
        "Monks-1": (333, None),               # 556
        # Additional complex OpenML datasets
        "Adult Census Income": (1590, 50000),
        "Credit-g": (31, None),
        "Bank Marketing": (1461, 30000),
        "Electricity": (151, 30000),
        "Phoneme": (1489, None),
        "Satimage": (182, None),
        "Madelon": (1485, None),
        "Amazon Employee Access": (4135, None),
    }


def get_sklearn_fetched_classification_specs() -> Dict[str, Dict[str, Optional[int]]]:
    """Map sklearn-fetched datasets to loader options."""
    return {
        "Covertype": {"max_rows": 100000},
        "KDDCup99": {"max_rows": 50000},
    }


def _cap_rows(
    X: Any,
    y: Any,
    max_rows: Optional[int],
    random_state: int,
) -> Tuple[Any, Any]:
    if max_rows is None or len(X) <= max_rows:
        return X, y

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=max_rows, replace=False)

    if hasattr(X, "iloc"):
        X = X.iloc[idx]
    else:
        X = np.asarray(X)[idx]

    if hasattr(y, "iloc"):
        y = y.iloc[idx]
    else:
        y = np.asarray(y)[idx]

    return X, y


def _preprocess_and_split(
    X: Any,
    y: Any,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_arr = np.asarray(y)

    y_enc = LabelEncoder().fit_transform(y_arr)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in numeric_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    X_train_out = pre.fit_transform(X_train)
    X_test_out = pre.transform(X_test)

    if hasattr(X_train_out, "toarray"):
        X_train_out = X_train_out.toarray()
    if hasattr(X_test_out, "toarray"):
        X_test_out = X_test_out.toarray()

    X_train_out = np.asarray(X_train_out, dtype=np.float32)
    X_test_out = np.asarray(X_test_out, dtype=np.float32)

    return X_train_out, X_test_out, y_train, y_test


def load_openml_classification_dataset(
    data_id: int,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True)
    X, y = _cap_rows(X=X, y=y, max_rows=max_rows, random_state=random_state)

    return _preprocess_and_split(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
    )


def load_sklearn_builtin_classification_dataset(
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    builtins = {
        "Iris": load_iris(return_X_y=True),
        "Wine": load_wine(return_X_y=True),
        "Breast Cancer": load_breast_cancer(return_X_y=True),
        "Digits": load_digits(return_X_y=True),
    }

    if name not in builtins:
        raise ValueError(f"Unknown sklearn builtin dataset: {name}")

    X, y = builtins[name]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_sklearn_fetched_classification_dataset(
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name == "Covertype":
        X, y = fetch_covtype(return_X_y=True)
    elif name == "KDDCup99":
        X, y = fetch_kddcup99(return_X_y=True, subset="SA", percent10=True)
    else:
        raise ValueError(f"Unknown sklearn fetched dataset: {name}")

    X_df = pd.DataFrame(X)
    y_arr = np.asarray(y)

    # Decode byte-encoded values for consistent preprocessing.
    for col in X_df.columns:
        if X_df[col].dtype == object:
            X_df[col] = X_df[col].map(
                lambda v: v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v
            )

    if y_arr.dtype == object or y_arr.dtype.kind in {"S", "U"}:
        y_arr = np.array(
            [
                v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v
                for v in y_arr
            ]
        )

    X_df, y_arr = _cap_rows(X=X_df, y=y_arr, max_rows=max_rows, random_state=random_state)

    return _preprocess_and_split(
        X=X_df,
        y=y_arr,
        test_size=test_size,
        random_state=random_state,
    )


def load_classification_datasets(
    dataset_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    logging: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    openml_specs = get_openml_classification_specs()
    sklearn_fetched_specs = get_sklearn_fetched_classification_specs()
    split_datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for name in dataset_names:
        if name in {"Iris", "Wine", "Breast Cancer", "Digits"}:
            split_datasets[name] = load_sklearn_builtin_classification_dataset(
                name=name,
                test_size=test_size,
                random_state=random_state,
            )
            if logging:
                print(f"Loaded builtin dataset: {name}")
            continue

        if name in openml_specs:
            data_id, max_rows = openml_specs[name]
            split_datasets[name] = load_openml_classification_dataset(
                data_id=data_id,
                test_size=test_size,
                random_state=random_state,
                max_rows=max_rows,
            )
            if logging:
                print(f"Loaded OpenML dataset: {name} (cap={max_rows})")
            continue

        if name in sklearn_fetched_specs:
            max_rows = sklearn_fetched_specs[name]["max_rows"]
            split_datasets[name] = load_sklearn_fetched_classification_dataset(
                name=name,
                test_size=test_size,
                random_state=random_state,
                max_rows=max_rows,
            )
            if logging:
                print(f"Loaded sklearn fetched dataset: {name} (cap={max_rows})")
            continue

        raise ValueError(f"Dataset '{name}' not supported.")

    return split_datasets


def load(
    dataset_names: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    logging: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convenience function to load datasets.

    If dataset_names is None or empty, loads all available datasets.
    """
    if dataset_names is None or len(dataset_names) == 0:
        dataset_names = get_all_dataset_names()

    return load_classification_datasets(
        dataset_names=dataset_names,
        test_size=test_size,
        random_state=random_state,
        logging=logging,
    )


def _run_benchmark(
    model: Any,
    dataset_names: List[str],
) -> Dict[str, float]:
    datasets = load_classification_datasets(
        dataset_names=dataset_names,
        test_size=0.2,
        random_state=42,
        logging=True,
    )

    scores: Dict[str, float] = {}

    for dataset_name, (X_train, X_test, y_train, y_test) in datasets.items():
        t0 = time.perf_counter()

        model.fit(np.asarray(X_train), y_train)
        y_pred = model.predict(np.asarray(X_test))

        score = accuracy_score(y_test, y_pred)
        dt = time.perf_counter() - t0

        scores[dataset_name] = float(score)
        print(f"{dataset_name:24s}  acc={score:.4f}  time={dt:.2f}s")

    return scores


def test_on_infinity_benchmark(model: Any) -> Dict[str, float]:
    """
    Test a model on the fixed Infinity Benchmark (the original 20 datasets).
    """
    return _run_benchmark(
        model=model,
        dataset_names=get_infinity_dataset_names(),
    )


def test_on_all_datasets(model: Any) -> Dict[str, float]:
    """
    Test a model on all datasets available in this package.
    """
    return _run_benchmark(
        model=model,
        dataset_names=get_all_dataset_names(),
    )


def test_on_datasets(model: Any, dataset_names: List[str]) -> Dict[str, float]:
    """
    Test a model on a specific list of dataset names.
    """
    if dataset_names is None or len(dataset_names) == 0:
        raise ValueError("dataset_names must be a non-empty list.")

    return _run_benchmark(
        model=model,
        dataset_names=dataset_names,
    )
