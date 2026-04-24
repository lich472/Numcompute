import numpy as np
import time


def benchmark_mse(n=100000, repeats=5):
    """
    Compare loop vs vectorised MSE.

    Parameters
    ----------
    n : int
        Number of samples.
    repeats : int
        Number of repetitions for averaging.

    Returns
    -------
    dict
        Average execution times and speedup.
    """
    y_true = np.random.rand(n)
    y_pred = np.random.rand(n)

    loop_times = []
    vec_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        mse_loop = 0.0
        for i in range(n):
            mse_loop += (y_true[i] - y_pred[i]) ** 2
        mse_loop /= n
        loop_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        mse_vec = np.mean((y_true - y_pred) ** 2)
        vec_times.append(time.perf_counter() - start)

    loop_time = np.mean(loop_times)
    vec_time = np.mean(vec_times)

    return {
        "loop_time": loop_time,
        "vectorised_time": vec_time,
        "speedup": loop_time / vec_time if vec_time > 0 else np.inf
    }


def benchmark_accuracy(n=100000, repeats=5):
    """
    Compare loop vs vectorised accuracy.

    Parameters
    ----------
    n : int
        Number of samples.
    repeats : int
        Number of repetitions for averaging.

    Returns
    -------
    dict
        Average execution times and speedup.
    """
    y_true = np.random.randint(0, 2, size=n)
    y_pred = np.random.randint(0, 2, size=n)

    loop_times = []
    vec_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        correct = 0
        for i in range(n):
            if y_true[i] == y_pred[i]:
                correct += 1
        acc_loop = correct / n
        loop_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        acc_vec = np.mean(y_true == y_pred)
        vec_times.append(time.perf_counter() - start)

    loop_time = np.mean(loop_times)
    vec_time = np.mean(vec_times)

    return {
        "loop_time": loop_time,
        "vectorised_time": vec_time,
        "speedup": loop_time / vec_time if vec_time > 0 else np.inf
    }


def run_all_benchmarks():
    """
    Run all benchmarks and print results.
    """
    print("=== Benchmark Results ===")

    mse_results = benchmark_mse()
    print("\nMSE Benchmark:")
    for k, v in mse_results.items():
        print(f"{k}: {v:.10f}")

    acc_results = benchmark_accuracy()
    print("\nAccuracy Benchmark:")
    for k, v in acc_results.items():
        print(f"{k}: {v:.10f}")