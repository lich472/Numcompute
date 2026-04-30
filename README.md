# NumCompute

NumCompute is a modular machine learning framework built from scratch using **Python and NumPy only**.
It replicates core functionality of libraries like scikit-learn, focusing on **numerical computation, vectorisation, and clean software design**.

---

## Features

* CSV data loading with missing value handling
* Data preprocessing (StandardScaler, MinMaxScaler, OneHotEncoder)
* Sorting and searching algorithms (top-k, quickselect, binary search)
* Ranking with tie handling and percentile computation
* Statistical functions (mean, median, std, histogram, quantiles)
* Evaluation metrics (accuracy, precision, recall, F1, MSE, confusion matrix)
* Numerical optimisation (finite-difference gradient and Jacobian)
* Pipeline abstraction for chaining transformations
* Utility functions (activations, distances, batching, logsumexp)
* Benchmarking (loop vs vectorised performance comparison)

---

## Project Structure

```text
NumCompute/
├── numcompute/
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stat.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_sort_search.py
│   ├── test_rank.py
│   ├── test_stat.py
│   ├── test_metrics.py
│   ├── test_optim.py
│   ├── test_pipeline.py
│   ├── test_utils.py
│   └── test_benchmarking.py
├── demo/
│   └── quickstart.ipynb
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <https://github.com/lich472/Numcompute.git>
cd Numcompute
```

Optional (recommended for clean imports):

```bash
pip install -e .
```

---

## Running the Demo

### Option 1 (Recommended)

```bash
cd demo
jupyter notebook quickstart.ipynb
```

### Option 2 (Install and run)

```bash
pip install -e .
```

Then run the notebook from anywhere.

---

## Running Tests

```bash
python -m unittest discover -s tests
```

Tests cover:

* core algorithms
* edge cases (NaN values, ties, duplicates)
* integration between modules

---

## Benchmark Results

| Metric   | Loop Time (s) | Vectorised Time (s) | Speedup |
| -------- | ------------- | ------------------- | ------- |
| MSE      | ~0.03         | ~0.001              | ~30x    |
| Accuracy | ~0.02         | ~0.0004             | ~50x    |

Vectorised implementations provide significant performance improvements over loop-based methods.

---

## Design Highlights

* Fully vectorised computations using NumPy
* Numerical stability (logsumexp, stable softmax, NaN handling)
* Modular architecture with consistent API
* Clear input/output shape handling and validation

---

## Notes

* No external ML/DL libraries were used
* Focus on algorithmic implementation and understanding
* Designed to simulate real-world ML library structure

---

## Usage Note

The demo notebook is designed to run within the project structure.
If running independently, ensure the package is installed using:

```bash
pip install -e .
```

---
