import os
import sys
import numpy as np

# fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from numcompute.io import read_csv
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from numcompute.pipeline import Pipeline
from numcompute.rank import rank, percentile
from numcompute.sort_search import stable_sort, multi_key_sort, topk, quickselect, binary_search
from numcompute.stat import mean, median, std, histogram, quantiles
from numcompute.metrics import accuracy, precision, recall, f1, mse, confusion_matrix
from numcompute.optim import grad, jacobian
from numcompute.benchmarking import run_all_benchmarks


def main():
    print("\n=== NumCompute Full Demo ===")

    # -----------------------------
    # 1. CSV (io.py)
    # -----------------------------
    file_path = "demo/sample.csv"

    with open(file_path, "w") as f:
        f.write("age,study,score\n")
        f.write("18,5,70\n")
        f.write("19,6,80\n")
        f.write("20,,90\n")
        f.write("21,8,85\n")

    X = read_csv(file_path, return_dict=False)
    print("\nCSV Loaded:")
    print(X)

    # -----------------------------
    # 2. Preprocessing
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\nStandardScaler:")
    print(X_scaled)

    minmax = MinMaxScaler()
    X_minmax = minmax.fit_transform(X)
    print("\nMinMaxScaler:")
    print(X_minmax)

    categories = np.array(["A", "B", "A", "C"])
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(categories)
    print("\nOneHotEncoder:")
    print(encoded)

    # -----------------------------
    # 3. Pipeline
    # -----------------------------
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("minmax", MinMaxScaler())
    ])

    X_pipe = pipe.fit_transform(X)
    print("\nPipeline Output:")
    print(X_pipe)

    # -----------------------------
    # 4. Stats
    # -----------------------------
    print("\nStats:")
    X_clean = np.nan_to_num(X, nan=0.0)

    print("Mean:", mean(X_clean))
    print("Median:", median(X_clean))
    print("Std:", std(X_clean))
    edges, counts = histogram(X[:, 2], bins=3)  # use only score column
    print("Histogram counts:", counts)
    print("Quantiles (0.5):", quantiles(X, 0.5))

    # -----------------------------
    # 5. Rank + Percentile
    # -----------------------------
    scores = np.array([70, 90, 80, 80])
    print("\nRanking:")
    print("Average rank:", rank(scores, "average"))
    print("Dense rank:", rank(scores, "dense"))
    print("90th percentile:", percentile(scores, 90))

    # -----------------------------
    # 6. Sort & Search
    # -----------------------------
    arr = np.array([10, 50, 20, 40])
    print("\nSort & Search:")
    print("Sorted:", stable_sort(arr))
    print("Top 2:", topk(arr, 2, largest=True, return_indices=False))
    print("2nd smallest:", quickselect(arr, 1))
    print("Binary search (40):", binary_search(np.array([10, 20, 40, 50]), 40))

    table = np.array([
        [2, 3],
        [1, 5],
        [1, 2],
        [2, 1]
    ])
    print("Multi-key sort:")
    print(multi_key_sort(table, keys=[0, 1]))

    # -----------------------------
    # 7. Metrics
    # -----------------------------
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    print("\nMetrics:")
    print("Accuracy:", accuracy(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1:", f1(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("MSE:", mse(np.array([1, 2, 3]), np.array([1, 2, 4])))

    # -----------------------------
    # 8. Optim (Grad/Jacobian)
    # -----------------------------
    def f(x):
        return x[0]**2 + x[1]**2

    print("\nGradient:", grad(f, np.array([3.0, 4.0])))

    def F(x):
        return np.array([x[0] + x[1], x[0] * x[1]])

    print("Jacobian:\n", jacobian(F, np.array([2.0, 3.0])))

    # -----------------------------
    # 9. Benchmarking
    # -----------------------------
    print("\nBenchmark:")
    run_all_benchmarks()


if __name__ == "__main__":
    main()