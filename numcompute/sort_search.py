# ### `sort_search.py`
# - **Sorting**:
#   - Stable sort wrapper (`np.sort(kind='stable')`)
#   - Multi-key sort (sort by multiple columns)
# - **Top-k / Partial Sort**:
#   - `topk(values, k, largest=True, return_indices=True)` using `np.argpartition`
#   - Implement **quickselect** for educational purposes
# - **Searching**:
#   - `binary_search(sorted_array, x)` returning insertion index and existence boolean

import numpy as np
import bisect

def stable_sort(arr):
    sorted_array = np.sort(arr, kind='stable')
    return sorted_array

#Pseducode:
# need to know how many keys(column) in input 
# Note: should sort by least important key first and then the most important key  
# using np.lexsort() to sort with multiple keys  

def multi_key_sort_Numpy(arr: np.ndarray, keys: list[str]): # the keys list is the priority order of index of column (the least --> the most important)
    for i in keys:
        sort_key_arr = np.sort(arr, kind='stable', order=keys[i])
        arr = sort_key_arr # Update the new array 
    return sort_key_arr
# column students : (name, grade)
dtype=[('name','U10'),('grade',int)]
students = np.array([("Alice", 90), ("Bob", 90), ("Charlie", 85)], dtype = dtype)

print(multi_key_sort_Numpy(students,['name','grade']))



# Instead od using "for" loop --> better approach using ranking by vectorisation ?

# - **Top-k / Partial Sort**:
#   - `topk(values, k, largest=True, return_indices=True)` using `np.argpartition`
#   - Implement **quickselect** for educational purposes
def topk(values, k, largest=True, return_indices=True):
    top_k_idx_partial_largest = np.argpartition(values, -k)[-k:]
    top_k_idx_partial_smallest = np.argpartition(values, k)[:k]
    top_k_vals_partial_largest = values[top_k_idx_partial_largest]
    top_k_vals_partial_smallest = values[top_k_idx_partial_smallest]
    if largest:
        if return_indices:
            return top_k_idx_partial_largest
        else:
            return top_k_vals_partial_largest
    else:
        if not return_indices:
            return top_k_vals_partial_smallest
        else:
            return top_k_idx_partial_smallest
        
def partition(arr, start_idx, end_idx): # using Lomuto's Partitioning
    pivot = arr[end_idx] # choose the pivot is the last right of arr
    i = start_idx
    for j in range(start_idx, end_idx): 
        if(arr[j]<pivot):
            temp = arr[i] # Swap the arr[j] with arr[i]
            arr[i] = arr[j]
            arr[j] = temp
            i+=1
    # swap the new position of pivot, arr[i] with pivot
    temp = arr[i]
    arr[i] = arr[end_idx]
    arr[end_idx] = temp

    pivot_idx = i
    return pivot_idx

def quick_search(arr, kth_smallest):
    start_idx = 0
    end_idx = len(arr)-1
    kth_smallest -=1 # user-friendly -> change to natural think
    #loop
    while True:
        if(start_idx == end_idx):
            return arr[start_idx]
        idx_pivot = partition(arr,start_idx,end_idx) # return idx of pivot
        if(idx_pivot == kth_smallest):
            return arr[idx_pivot]
        elif(kth_smallest > idx_pivot): # start from right of pivot_idx
            start_idx = idx_pivot + 1
        elif(kth_smallest < idx_pivot): # start from the left of pivot_idx
            end_idx = idx_pivot - 1
scores = np.array([1, 3, 7, 2, 5, 8, 10])
print(partition(scores, 0, len(scores)-1))
print(quick_search(scores,5))

