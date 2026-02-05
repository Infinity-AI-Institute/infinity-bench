# Infinity Benchmark

A comprehensive benchmark suite for classification algorithms with 20 diverse datasets.

## Overview

This repository provides a standardized dataset loader and baseline benchmarking tools for evaluating classification models across 20 carefully selected datasets from scikit-learn and OpenML:

### Datasets

**Built-in (scikit-learn):**
- Iris
- Wine
- Breast Cancer
- Digits

**OpenML Datasets:**
- Balance Scale
- Blood Transfusion
- Haberman
- Seeds
- Teaching Assistant
- Zoo
- Planning Relax
- Ionosphere
- Sonar
- Glass
- Vehicle
- Liver Disorders
- Heart Statlog
- Pima Indians Diabetes
- Australian
- Monks-1

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Load Datasets

```python
from data_loader import load

# Load all 20 datasets
datasets = load()

# Load specific datasets
datasets = load(["Iris", "Wine", "Breast Cancer"])
```

### Benchmark Your Model

```python
from data_loader import test_on_infinity_benchmark

model = YourModel()  # Your model here

# Test on all 20 datasets
scores = test_on_infinity_benchmark(model)

# Test on specific datasets
scores = test_on_infinity_benchmark(model, ["Iris", "Wine"])

# Get just the scores without verbose output
scores = test_on_infinity_benchmark(model, verbose=False)
```

### Accessing specific datastes

```python
from data_loader import load_classification_datasets

# Load all 20 datasets
datasets = load_classification_datasets([
    "Iris", "Wine", "Breast Cancer", "Digits",
    "Balance Scale", "Blood Transfusion", "Haberman", "Seeds",
    "Teaching Assistant", "Zoo", "Planning Relax", "Ionosphere",
    "Sonar", "Glass", "Vehicle", "Liver Disorders",
    "Heart Statlog", "Pima Indians Diabetes", "Australian", "Monks-1"
], logging=True)

# Access a specific dataset
X_train, X_test, y_train, y_test = datasets["Iris"]
```

### Manual Per Dataset Benchmarking 

```python
from data_loader import load_classification_datasets
from sklearn.metrics import accuracy_score
import numpy as np

datasets = load_classification_datasets([...], logging=True)

for dataset_name, (X_train, X_test, y_train, y_test) in datasets.items():
    model = YourModel()  # Your model here
    model.fit(np.asarray(X_train), y_train)
    y_pred = model.predict(np.asarray(X_test))
    score = accuracy_score(y_test, y_pred)
    print(f"{dataset_name}: {score:.4f}")
```

## API Reference

### `load(dataset_names=None, test_size=0.2, random_state=42, logging=False)`

Load datasets from the Infinity Benchmark.

**Parameters:**
- `dataset_names` (list, optional): Names of datasets to load. If None or empty, loads all 20 datasets.
- `test_size` (float): Train/test split ratio (default: 0.2)
- `random_state` (int): Random seed (default: 42)
- `logging` (bool): Print dataset loading info (default: False)

**Returns:** Dictionary mapping dataset names to (X_train, X_test, y_train, y_test) tuples

### `test_on_infinity_benchmark(model, dataset_names=None, test_size=0.2, random_state=42, verbose=True)`

Test a model on the Infinity Benchmark datasets.

**Parameters:**
- `model`: A scikit-learn compatible model with `.fit()` and `.predict()` methods
- `dataset_names` (list, optional): Names of datasets to test on. If None or empty, tests all 20.
- `test_size` (float): Train/test split ratio (default: 0.2)
- `random_state` (int): Random seed (default: 42)
- `verbose` (bool): Print results for each dataset (default: True)

**Returns:** Dictionary mapping dataset names to accuracy scores

## Notes

- The built-in datasets (Iris, Wine, etc.) are downloaded automatically by scikit-learn
- OpenML datasets are fetched on first use and cached locally
- All data is converted to float32 for numerical stability
- Missing features/categorical variables are handled automatically
