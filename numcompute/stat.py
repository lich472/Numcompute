import numpy as np
from rank import percentile
def mean(arr, axis=None, keepdims=False):
    sum = np.sum(arr, axis = axis, keepdims=keepdims)
    if(axis is None):
        mean = sum/(arr.size)
        return mean
    else:
        mean = sum/arr.shape[axis]
        return mean, mean.shape

def median(arr, axis=None):
    arr = np.array(arr)
    sorted_arr = np.sort(arr, axis=axis, kind='stable')
    if(axis==None): # flat array into 1D and median 
        n = sorted_arr.size
        if(n%2==0): #if even length -> add 2 middle /2 , odd length ->middle
            lower_idx = n//2 -1
            higher_idx = n//2
            median = (sorted_arr[lower_idx] + sorted_arr[higher_idx])/2
            return median
        else:
            mid_idx = n//2
            median = sorted_arr[mid_idx]
            return median, median.shape
    # work along axis
    n = sorted_arr.shape[axis]
    mid_idx = n//2
    if(n%2==0):
        lower = np.take(sorted_arr, mid_idx - 1, axis=axis)
        higher = np.take(sorted_arr, mid_idx, axis=axis)
        median = (lower + higher)/2
        return median, median.shape
    else:
        median = np.take(sorted_arr, mid_idx, axis=axis)
        return median, median.shape

def std(arr, axis=None, ddof=0): # ddof supports for both Popular (ddof = 0) and Sample (ddof = 1) standard deviation 
    arr = np.array(arr)
    # need mean() method first
    arr_mean = mean(arr, axis = axis, keepdims=True) # using keepdims here to preserves the collapsed dimension as size 1, making broadcasting work correctly in all cases.
    deviation = np.subtract(arr,arr_mean)
    squared = deviation**2
    if(axis is None):
        n=arr.size
    else:
        n=arr.shape[axis]
    variance = np.sum(squared, axis=axis) / (n - ddof)
    std = np.sqrt(variance)
    return std

def min(arr, axis=None):
    arr=np.array(arr)
    if( not np.issubdtype(arr.dtype, np.number) ):
        raise ValueError("Only numerics array is allowed.")
    if(arr.size == 0):
        raise ValueError("Empty array is not allowed.")
    if axis is not None and not isinstance(axis, int):
        raise TypeError("integer argument expected")
    if axis is not None and not (-arr.ndim <= axis < arr.ndim):
        raise ValueError(f"axis {axis} is out of bounds for array with {arr.ndim} dimensions")
    val=np.min(arr,axis=axis)
    return val


def max(arr, axis=None):
    arr=np.array(arr)
    if( not np.issubdtype(arr.dtype, np.number) ):
        raise ValueError("Only numerics array is allowed.")
    if(arr.size == 0):
        raise ValueError("Empty array is not allowed.")
    if axis is not None and not isinstance(axis, int):
        raise TypeError("integer argument expected")
    if axis is not None and not (-arr.ndim <= axis < arr.ndim):
        raise ValueError(f"axis {axis} is out of bounds for array with {arr.ndim} dimensions")
    val=np.max(arr,axis=axis)
    return val

def histogram(arr, bins=10):
    arr=np.array(arr)
    flat_arr=arr.ravel()
    if( not np.issubdtype(flat_arr.dtype, np.number) ):
        raise ValueError("Only numerics array is allowed.")
    if(flat_arr.size == 0):
        raise ValueError("Empty array is not allowed.")
    if( not bins > 0 ):
        raise ValueError("Only positive bins is allowed.")
    min_val = min(flat_arr)
    max_val = max(flat_arr)
    edges = np.linspace(min_val, max_val, bins + 1 ) # bins + 1 as if bins = 5 -> need 6 points
    val = np.searchsorted(edges, flat_arr, side='right') - 1
    bin_indices = np.clip(val, 0, bins - 1)
    count = np.bincount(bin_indices, minlength=bins) 
    return edges, count

def quantiles(data, q, interpolation='linear'):
    arr = np.array(data)
    q_arr=np.array(q)
    if( not np.issubdtype(arr.dtype, np.number) ):
        raise ValueError("Only numerics array is allowed.")
    if(np.any(q_arr<0)or np.any(q_arr>1)):
        raise ValueError("q should be from 0 to 1")
    mask = np.isnan(arr)
    clean_arr = arr[~mask]
    if(clean_arr.size == 0):
        raise ValueError("Empty array is not allowed.")
    else:
        result = percentile(clean_arr, q*100,interpolation=interpolation)
        return result