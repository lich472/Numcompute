# NumCompute-Stream: Streaming ML Framework from Scratch

This is a custom, modular Python framework built entirely using **NumPy** and **matplotlib** to simulate online/streaming machine learning. It supports incremental data ingestion, pipelined preprocessing, performance logging, real-time evaluation, and tree-based ensemble models.

---

## Project Structure

*   `numcompute_stream/` - Core package directory
    *   `io.py` - File reader that yields data chunks sequentially.
    *   `preprocessing.py` - `StandardScaler` that updates its mean and variance incrementally.
    *   `tree.py` - Custom `DecisionTreeClassifier` with `partial_fit` capabilities.
    *   `ensemble.py` - `RandomForestClassifier` ensemble built on top of our tree class.
    *   `pipeline.py` - Chains transformers and models together.
    *   `metrics.py` - Tracks classification scores (`accuracy`, `f1`, `auc`, `confusion_matrix`) over chunks using a rolling window.
    *   `stream.py` - `StreamTrainer` which orchestrates chunk training, testing, and performance/memory logging.
    *   `visualise.py` - Helper module using matplotlib to generate evaluation plots.
*   `tests/` - Folder containing comprehensive unit tests covering standard execution paths and edge cases.
*   `benchmark/` - Contains the script comparing loop-heavy versus vectorized math.
*   `demo/` - `stream_demo.ipynb` Jupyter Notebook showing a complete execution run.

---

## Installation

Clone the repository:

```bash
git clone <https://github.com/lich472/Numcompute_stream.git>
cd Numcompute
```

Optional (recommended for clean imports):

```bash
pip install -e .
```

---

## How to Run the Code

### 1. Run the Unit Tests
To run all tests and verify that everything functions correctly across the entire package, open your terminal in the root folder and run:
```bash
python -m unittest discover -s tests
```

### 2. Run the Speed Benchmark
To see how much faster our vectorized NumPy implementation is compared to standard Python loops, run the benchmark script:
```bash
python benchmark/speed_benchmark.py   
```
--- Running Loop vs Vectorized Benchmark ---
Loop implementation time: 0.4180 seconds
Vectorized implementation time: 0.0356 seconds
Result: Vectorized code is 11.7x faster!

### 3. Run the Streaming Demo Notebook
To see the full framework in action on the Iris dataset:
1. Navigate to the `demo/` folder.
2. Open `stream_demo.ipynb` in Jupyter Notebook or VS Code.
3. Run the cells sequentially to see the model download data, read chunks, update metrics, and display performance graphs.

---

## Key Implementation Notes

*   **Streaming Strategy (`partial_fit`):** For non-parametric estimators like decision trees and random forests, modifying an existing structure online is highly unstable. To handle streaming correctly, our framework accumulates chunk history sequentially and rebuilds the trees upon new data arrivals, matching real-world stream simulation patterns.
*   **Numerical Stability:** Handled missing/empty data using `np.isnan()`, `np.nanmean()`, and built-in type conversions to prevent runtime script drops during floating-point division or integer label casting on unaligned final file lines.
*   **Vectorization vs Loops:** Heavily relied on NumPy indexing configurations and vectorized sum utilities to replace slow multi-level Python nesting loops in complex bottleneck logic routines like impurity scoring.
