import time
import numpy as np
from tree import DecisionTreeClassifier

def naive_loop_impurity(y):
    if len(y) == 0:
        return 0.0
    
    # Simple manual count using a basic python dictionary loop
    counts = {}
    for label in y:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
        
    total = len(y)
    gini_sum = 0.0
    for label in counts:
        p = counts[label] / total
        gini_sum += p ** 2
        
    return 1.0 - gini_sum

def run_benchmark():
    print("--- Running Loop vs Vectorized Benchmark ---")
    
    # Set seed and create a random large array to test speed difference
    np.random.seed(42)
    y_large = np.random.choice([0, 1, 2], size=50000)
    
    # First, test the slow loop version
    t0 = time.time()
    for i in range(100):
        naive_loop_impurity(y_large)
    loop_time = time.time() - t0
    print(f"Loop implementation time: {loop_time:.4f} seconds")
    
    # Second, test our vectorized code from tree.py
    tree = DecisionTreeClassifier(criterion='gini')
    
    t1 = time.time()
    for i in range(100):
        tree.impurity(y_large)
    vectorized_time = time.time() - t1
    print(f"Vectorized implementation time: {vectorized_time:.4f} seconds")
    
    # calculate performance improvement factor
    speedup = loop_time / vectorized_time
    print(f"Result: Vectorized code is {speedup:.1f}x faster!")

if __name__ == "__main__":
    run_benchmark()
