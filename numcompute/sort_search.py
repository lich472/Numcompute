import numpy as np
import bisect

def stable_sort(arr):
    sorted_array = np.sort(arr, kind='stable')
    return sorted_array

#Pseducode:
# need to know how many keys(column) in input 
# Note: should sort by least important key first and then the most important key  
# using np.lexsort() to sort with multiple keys  

def multi_key_sort(arr: np.ndarray, keys: list[int], ascending = True):
    keys_col = [arr[:,k] for k in reversed(keys)] # need to reverse the keys as the lexsort, it sort from right to left 
    ranks=[np.unique(col, return_inverse = True)[1] for col in keys_col] # cannot use double np.argsort(np.argsort()) as it only supports for numerics arr
    if not ascending:
        ranks = [rank*(-1) for rank in ranks] # reverse rank to descending order
    ranks_tuple = tuple(ranks)
    sorted_arr = arr[np.lexsort(ranks_tuple)] 
    return sorted_arr

def topk(values: np.ndarray, k, largest=True, return_indices=True):
    arr = np.array(values)
    if largest:
        top_k_idx_partial_largest = np.argpartition(arr, -k)[-k:]
        top_k_vals_partial_largest = arr[top_k_idx_partial_largest]
        if return_indices:
            return top_k_idx_partial_largest
        else:
            return top_k_vals_partial_largest
    else:
        top_k_idx_partial_smallest = np.argpartition(arr, k)[:k]
        top_k_vals_partial_smallest = arr[top_k_idx_partial_smallest]
        if return_indices:
            return top_k_idx_partial_smallest
        else:
            return top_k_vals_partial_smallest
        
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
    print(arr) # for debug purpose
    return pivot_idx

def quickselect(data, kth_smallest):
    arr = np.array(data)
    if arr.size == 0:
        raise ValueError("Empty array is not allowed.")
    if kth_smallest < 1 or kth_smallest > len(arr):
        raise ValueError(f"k must be between 1 and {len(arr)}")
    if( not np.issubdtype(arr.dtype, np.number) ):
        raise ValueError("Only numerics array is allowed.")
    if( arr.ndim != 1):
        raise ValueError("Only 1D array is allowed.")
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

def binary_search(arr, target, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low > high:
        return (low, False)
    mid = (low + high) // 2
    if arr[mid] == target:
        return (mid, True)
    elif arr[mid] > target:
        return binary_search(arr, target, low, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, high)