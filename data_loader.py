# infinity-benchmark/data_loader.py

from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
import time

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    fetch_openml,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_all_dataset_names() -> List[str]:
    """Get all available dataset names (20 datasets)."""
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


def get_openml_classification_ids() -> Dict[str, Tuple[int, Optional[int]]]:
    return {
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
    }



def load_openml_classification_dataset(
    data_id: int,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True)

    if max_rows is not None and len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=random_state)
        y = y.loc[X.index]

    y = y.astype("category")
    y_enc = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
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
        sparse_threshold=0.0,  # force dense
    )


    X_train_out = pre.fit_transform(X_train)
    X_test_out = pre.transform(X_test)

    # ensure dense
    if hasattr(X_train_out, "toarray"):
        X_train_out = X_train_out.toarray()
    if hasattr(X_test_out, "toarray"):
        X_test_out = X_test_out.toarray()

    # ensure numeric dtype for all models
    X_train_out = np.asarray(X_train_out, dtype=np.float32)
    X_test_out = np.asarray(X_test_out, dtype=np.float32)


    return X_train_out, X_test_out, y_train, y_test


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


def load_classification_datasets(
    dataset_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    logging: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    openml_ids = get_openml_classification_ids()
    split_datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for name in dataset_names:
        # sklearn built-ins
        if name in {"Iris", "Wine", "Breast Cancer", "Digits"}:
            split_datasets[name] = load_sklearn_builtin_classification_dataset(
                name=name,
                test_size=test_size,
                random_state=random_state,
            )
            if logging:
                print(f"Loaded builtin dataset: {name}")
            continue

        # OpenML datasets (with caps)
        if name in openml_ids:
            data_id, max_rows = openml_ids[name]
            split_datasets[name] = load_openml_classification_dataset(
                data_id=data_id,
                test_size=test_size,
                random_state=random_state,
                max_rows=max_rows,
            )
            if logging:
                print(f"Loaded OpenML dataset: {name} (cap={max_rows})")
            continue

        raise ValueError(
            f"Dataset '{name}' not supported (builtin + OpenML only)."
        )

    return split_datasets


def load(
    dataset_names: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    logging: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convenience function to load datasets.
    
    If dataset_names is None or empty, loads all 20 datasets.
    
    Args:
        dataset_names: List of dataset names to load. If None or empty, loads all 20.
        test_size: Train/test split ratio (default: 0.2)
        random_state: Random seed (default: 42)
        logging: Whether to print logging info (default: False)
    
    Returns:
        Dictionary mapping dataset names to (X_train, X_test, y_train, y_test) tuples.
    
    Example:
        datasets = load()  # Load all 20 datasets
        datasets = load(["Iris", "Wine"])  # Load specific datasets
    """
    if dataset_names is None or len(dataset_names) == 0:
        dataset_names = get_all_dataset_names()
    
    return load_classification_datasets(
        dataset_names=dataset_names,
        test_size=test_size,
        random_state=random_state,
        logging=logging,
    )


def test_on_infinity_benchmark(
    model: Any,
    dataset_names: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Test a model on the Infinity Benchmark datasets.
    
    If dataset_names is None or empty, tests on all 20 datasets.
    
    Args:
        model: A scikit-learn compatible model with .fit() and .predict() methods.
        dataset_names: List of dataset names to test on. If None or empty, tests all 20.
        test_size: Train/test split ratio (default: 0.2)
        random_state: Random seed (default: 42)
        verbose: Whether to print results (default: True)
    
    Returns:
        Dictionary mapping dataset names to accuracy scores.
    
    Example:
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(random_state=42)
        scores = test_on_infinity_benchmark(model)
        # Or test on specific datasets:
        scores = test_on_infinity_benchmark(model, ["Iris", "Wine"])
    """
    if dataset_names is None or len(dataset_names) == 0:
        dataset_names = get_all_dataset_names()
    
    datasets = load_classification_datasets(
        dataset_names=dataset_names,
        test_size=test_size,
        random_state=random_state,
        logging=verbose,
    )
    
    scores: Dict[str, float] = {}
    
    for dataset_name, (X_train, X_test, y_train, y_test) in datasets.items():
        t0 = time.perf_counter()
        
        # Fit and predict
        model.fit(np.asarray(X_train), y_train)
        y_pred = model.predict(np.asarray(X_test))
        
        # Calculate accuracy
        score = accuracy_score(y_test, y_pred)
        dt = time.perf_counter() - t0
        
        scores[dataset_name] = float(score)
        
        if verbose:
            print(f"{dataset_name:24s}  acc={score:.4f}  time={dt:.2f}s")
    
    return scores
