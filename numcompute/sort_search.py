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

# def multi_key_sort_Numpy(arr: np.ndarray, keys: list[str]): # the keys list is the priority order of index of column (the least --> the most important)
#     for i in keys:
#         sort_key_arr = np.sort(arr, kind='stable', order=keys[i])
#         arr = sort_key_arr # Update the new array 
#     return sort_key_arr
# column students : (name, grade)

def multi_key_sort(arr: np.ndarray, keys: list[int], ascending = True):
    keys_column = tuple(arr[:,k] for k in reversed(keys)) # need to reverse the keys as the lexsort, it sort from right to left 
    if not ascending:
        # negate each collumn in tuple so satisfy the descensed sorting
        keys_column = tuple(col*(-1) for col in keys_column)
    print(keys_column)
    print(np.lexsort(keys_column))
    sorted_arr = arr[np.lexsort(keys_column)] # need to implement the "ascending or descendind sort concept"
    return sorted_arr

arr = np.array([[3, 6, 2], 
                [4, 2, 5], 
                [3, 3, 1]])


print(arr)
print(multi_key_sort(arr,[0,2], False)) 
print(multi_key_sort(arr,[0,1], True))


# TODO: After times (-1) which only work with numeric array --> using argsort + Ranking to be reflective (work on int + string)



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

