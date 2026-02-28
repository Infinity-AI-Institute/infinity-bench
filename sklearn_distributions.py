from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from data_loader import get_all_dataset_names
from get_distribution import EXCLUDED_DATASETS
from get_distribution import write_distribution_json

EstimatorFactory = Tuple[str, Callable[[], Any]]


CORE_ESTIMATOR_FACTORIES: List[EstimatorFactory] = [
    # Previously-run estimators:
    # ("sklearn.linear_model.LogisticRegression", lambda: LogisticRegression(max_iter=1000)),
    # ("sklearn.neighbors.KNeighborsClassifier", lambda: KNeighborsClassifier()),
    # ("sklearn.tree.DecisionTreeClassifier", lambda: DecisionTreeClassifier(random_state=42)),
    # ("sklearn.ensemble.RandomForestClassifier", lambda: RandomForestClassifier(random_state=42)),
    # ("sklearn.ensemble.GradientBoostingClassifier", lambda: GradientBoostingClassifier(random_state=42)),
    # ("sklearn.neural_network.MLPClassifier", lambda: MLPClassifier(max_iter=500, random_state=42)),
    # ("sklearn.linear_model.SGDClassifier", lambda: SGDClassifier(random_state=42)),
    ("sklearn.naive_bayes.GaussianNB", lambda: GaussianNB()),
    ("sklearn.discriminant_analysis.LinearDiscriminantAnalysis", lambda: LinearDiscriminantAnalysis()),
]


def _sanitize_name(qualname: str) -> str:
    return qualname.replace("sklearn.", "").replace(".", "_")


def run_core_estimators(
    output_dir: str = "sklearn_distribution_outputs",
    dataset_names: List[str] | None = None,
    freeze_splits: bool = True,
    freeze_dir: str = ".infinity_frozen_splits",
    refresh_frozen_splits: bool = False,
    logging: bool = True,
) -> Dict[str, Any]:
    if dataset_names is None:
        dataset_names = [name for name in get_all_dataset_names() if name not in EXCLUDED_DATASETS]

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "output_dir": str(out_root.resolve()),
        "freeze_splits": freeze_splits,
        "freeze_dir": freeze_dir,
        "refresh_frozen_splits": refresh_frozen_splits,
        "num_datasets": len(dataset_names),
        "datasets": dataset_names,
        "runs": [],
    }

    for qualname, factory in CORE_ESTIMATOR_FACTORIES:
        out_file = out_root / f"{_sanitize_name(qualname)}.json"
        run_info: Dict[str, Any] = {
            "estimator": qualname,
            "output_file": str(out_file),
            "status": "pending",
        }

        try:
            if logging:
                print(f"\nRunning {qualname}")
            model = factory()
            payload = write_distribution_json(
                model=model,
                output_path=str(out_file),
                dataset_names=dataset_names,
                logging=logging,
                freeze_splits=freeze_splits,
                freeze_dir=freeze_dir,
                refresh_frozen_splits=refresh_frozen_splits,
            )
            run_info["status"] = "ok"
            run_info["rows"] = len(payload["rows"])
            run_info["max_classes"] = int(payload["max_classes"])
        except Exception as exc:
            run_info["status"] = "error"
            run_info["error"] = f"{type(exc).__name__}: {exc}"
            if logging:
                print(f"Failed {qualname}: {run_info['error']}")

        summary["runs"].append(run_info)

    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if logging:
        ok_count = sum(1 for r in summary["runs"] if r["status"] == "ok")
        print(f"\nCompleted {ok_count}/{len(summary['runs'])} estimators.")
        print(f"Summary: {summary_path}")

    return summary


if __name__ == "__main__":
    run_core_estimators()
