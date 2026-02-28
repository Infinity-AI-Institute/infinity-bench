from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _extract_distribution(row: Dict[str, Any], target_dim: int) -> np.ndarray:
    class_keys = [k for k in row.keys() if k.startswith("class_") and k.endswith("_pred")]
    if not class_keys:
        raise ValueError("Row does not contain class_*_pred columns.")

    dist = np.zeros((target_dim,), dtype=np.float64)
    for key in class_keys:
        class_idx = int(key.split("_")[1])
        v = row[key]
        if 0 <= class_idx < target_dim and v is not None:
            dist[class_idx] = float(v)
    s = float(np.sum(dist))
    if s > 0:
        dist = dist / s
    return dist


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, eps=eps) + 0.5 * _kl_divergence(q, m, eps=eps)


def _load_model_rows(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return path.stem, payload["rows"]


def _validate_alignment(model_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    model_names = list(model_rows.keys())
    n_rows = len(model_rows[model_names[0]])

    for model_name in model_names:
        if len(model_rows[model_name]) != n_rows:
            raise ValueError("Model JSON files have different row counts.")

    base = model_rows[model_names[0]]
    for model_name in model_names[1:]:
        rows = model_rows[model_name]
        for i in range(n_rows):
            if (
                base[i].get("dataset") != rows[i].get("dataset")
                or int(base[i].get("real_class")) != int(rows[i].get("real_class"))
            ):
                raise ValueError(
                    f"Row alignment mismatch at row {i} between "
                    f"{model_names[0]} and {model_name}."
                )


def _infer_global_class_dim(model_rows: Dict[str, List[Dict[str, Any]]]) -> int:
    max_idx = -1
    for rows in model_rows.values():
        for row in rows:
            for key in row.keys():
                if key.startswith("class_") and key.endswith("_pred"):
                    class_idx = int(key.split("_")[1])
                    if class_idx > max_idx:
                        max_idx = class_idx
    if max_idx < 0:
        raise ValueError("No class_*_pred keys found in inputs.")
    return max_idx + 1


def _load_rows_from_dir(
    input_dir: str,
    include_datasets: Optional[List[str]] = None,
    include_models: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    root = Path(input_dir)
    json_paths = sorted(
        p for p in root.glob("*.json") if p.name != "summary.json" and "divergence" not in p.name
    )
    if not json_paths:
        raise ValueError(f"No model JSON files found in {input_dir}.")

    include_model_set = set(include_models) if include_models else None
    include_dataset_set = set(include_datasets) if include_datasets else None

    model_rows: Dict[str, List[Dict[str, Any]]] = {}
    for path in json_paths:
        model_name, rows = _load_model_rows(path)
        if include_model_set is not None and model_name not in include_model_set:
            continue
        if include_dataset_set is not None:
            rows = [r for r in rows if r.get("dataset") in include_dataset_set]
        if rows:
            model_rows[model_name] = rows

    if include_model_set is not None:
        missing = sorted(include_model_set - set(model_rows.keys()))
        if missing:
            raise ValueError(f"Requested models not found (or empty after filtering): {missing}")

    if not model_rows:
        raise ValueError("No rows available after model/dataset filtering.")

    return model_rows


def compute_leave_one_out_divergence(
    input_dir: str = "sklearn_distribution_outputs",
    output_path: str = "sklearn_distribution_outputs/divergence_vs_ensemble.json",
    include_datasets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    root = Path(input_dir)
    model_rows = _load_rows_from_dir(
        input_dir=input_dir,
        include_datasets=include_datasets,
    )
    if len(model_rows) < 2:
        raise ValueError("Need at least 2 model JSON files to compute leave-one-out divergence.")

    _validate_alignment(model_rows)
    global_class_dim = _infer_global_class_dim(model_rows)

    model_names = list(model_rows.keys())
    n_models = len(model_names)
    n_rows = len(model_rows[model_names[0]])

    # Precompute model distributions: [n_models, n_rows, n_classes]
    dists = []
    for model_name in model_names:
        rows = model_rows[model_name]
        dists.append(np.stack([_extract_distribution(r, target_dim=global_class_dim) for r in rows], axis=0))
    dist_tensor = np.stack(dists, axis=0)

    results: List[Dict[str, Any]] = []
    per_row_records: List[Dict[str, Any]] = []

    for m_idx, model_name in enumerate(model_names):
        others_mask = np.ones(n_models, dtype=bool)
        others_mask[m_idx] = False

        model_dist = dist_tensor[m_idx]  # [n_rows, n_classes]
        ensemble_dist = dist_tensor[others_mask].mean(axis=0)  # [n_rows, n_classes]

        row_kl = np.asarray(
            [_kl_divergence(model_dist[i], ensemble_dist[i]) for i in range(n_rows)],
            dtype=np.float64,
        )
        row_js = np.asarray(
            [_js_divergence(model_dist[i], ensemble_dist[i]) for i in range(n_rows)],
            dtype=np.float64,
        )

        results.append(
            {
                "model": model_name,
                "n_rows": int(n_rows),
                "mean_kl": float(np.mean(row_kl)),
                "median_kl": float(np.median(row_kl)),
                "mean_js": float(np.mean(row_js)),
                "median_js": float(np.median(row_js)),
            }
        )

        rows = model_rows[model_name]
        for i in range(n_rows):
            per_row_records.append(
                {
                    "model": model_name,
                    "row_index": i,
                    "dataset": rows[i]["dataset"],
                    "real_class": int(rows[i]["real_class"]),
                    "kl_vs_ensemble": float(row_kl[i]),
                    "js_vs_ensemble": float(row_js[i]),
                }
            )

    output = {
        "input_dir": str(root.resolve()),
        "models": model_names,
        "num_models": int(n_models),
        "num_rows": int(n_rows),
        "included_datasets": include_datasets or "all",
        "metric_notes": "KL(P||Q) and Jensen-Shannon divergence; Q is leave-one-out mean distribution.",
        "model_summary": sorted(results, key=lambda x: x["mean_js"], reverse=True),
        "per_row": per_row_records,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def compute_main_vs_selected_ensemble(
    main_model: str,
    compare_models: List[str],
    input_dir: str = "sklearn_distribution_outputs",
    output_path: str = "sklearn_distribution_outputs/divergence_main_vs_selected_ensemble.json",
    include_datasets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not compare_models:
        raise ValueError("compare_models must be non-empty.")
    if main_model in compare_models:
        raise ValueError("main_model should not be included in compare_models.")

    selected_models = [main_model] + compare_models
    model_rows = _load_rows_from_dir(
        input_dir=input_dir,
        include_datasets=include_datasets,
        include_models=selected_models,
    )
    _validate_alignment(model_rows)
    global_class_dim = _infer_global_class_dim(model_rows)

    main_rows = model_rows[main_model]
    n_rows = len(main_rows)

    main_dist = np.stack(
        [_extract_distribution(r, target_dim=global_class_dim) for r in main_rows],
        axis=0,
    )
    compare_dists = np.stack(
        [
            np.stack(
                [_extract_distribution(r, target_dim=global_class_dim) for r in model_rows[m]],
                axis=0,
            )
            for m in compare_models
        ],
        axis=0,
    )
    ensemble_dist = compare_dists.mean(axis=0)

    row_kl = np.asarray([_kl_divergence(main_dist[i], ensemble_dist[i]) for i in range(n_rows)], dtype=np.float64)
    row_js = np.asarray([_js_divergence(main_dist[i], ensemble_dist[i]) for i in range(n_rows)], dtype=np.float64)

    output = {
        "input_dir": str(Path(input_dir).resolve()),
        "main_model": main_model,
        "compare_models": compare_models,
        "included_datasets": include_datasets or "all",
        "num_rows": int(n_rows),
        "score": {
            "mean_kl": float(np.mean(row_kl)),
            "median_kl": float(np.median(row_kl)),
            "mean_js": float(np.mean(row_js)),
            "median_js": float(np.median(row_js)),
        },
        "metric_notes": "KL(P||Q) and Jensen-Shannon divergence; Q is the average distribution of compare_models.",
        "per_row": [
            {
                "row_index": i,
                "dataset": main_rows[i]["dataset"],
                "real_class": int(main_rows[i]["real_class"]),
                "kl_vs_ensemble": float(row_kl[i]),
                "js_vs_ensemble": float(row_js[i]),
            }
            for i in range(n_rows)
        ],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="sklearn_distribution_outputs")
    parser.add_argument("--output-path", default="sklearn_distribution_outputs/divergence_vs_ensemble.json")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names to include, e.g. 'Iris,Wine'. Empty means all datasets.",
    )
    parser.add_argument(
        "--main-model",
        default="",
        help="Model name (JSON stem) to score against an ensemble of selected models.",
    )
    parser.add_argument(
        "--compare-models",
        default="",
        help="Comma-separated model names (JSON stems) used to build the ensemble for --main-model.",
    )
    args = parser.parse_args()

    include = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets else None
    compare_models = [m.strip() for m in args.compare_models.split(",") if m.strip()] if args.compare_models else []

    if args.main_model:
        out = compute_main_vs_selected_ensemble(
            main_model=args.main_model.strip(),
            compare_models=compare_models,
            input_dir=args.input_dir,
            output_path=args.output_path,
            include_datasets=include,
        )
        print(f"Wrote main-vs-ensemble score for {out['main_model']} on {out['num_rows']} rows")
        print(
            f"mean_kl={out['score']['mean_kl']:.6f}, "
            f"median_kl={out['score']['median_kl']:.6f}, "
            f"mean_js={out['score']['mean_js']:.6f}"
        )
    else:
        out = compute_leave_one_out_divergence(
            input_dir=args.input_dir,
            output_path=args.output_path,
            include_datasets=include,
        )
        print(f"Wrote {out['num_models']} models x {out['num_rows']} rows")
        print("Top models by mean_js:")
        for item in out["model_summary"]:
            print(f"{item['model']}: mean_js={item['mean_js']:.6f}, mean_kl={item['mean_kl']:.6f}")
