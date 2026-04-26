def mean(arr, axis=None):
    sum = np.sum(arr, axis = axis)
    if(axis is None):
        mean = sum/(arr.size)
        return mean
    else:
        mean = sum/arr.shape[axis]
        return mean

def median(arr, axis=None):
    sorted_arr = np.sort(arr, axis=axis, kind='stable')
    if(axis==None): # flat array into 1D and median 
        n = sorted_arr.size
        if(n%2==0): #even -> #if even length -> add 2 middle /2 , odd length ->middle
            lower_idx = n//2 -1
            higher_idx = n//2
            median = (sorted_arr[lower_idx] + sorted_arr[higher_idx])/2
            return median
        else:
            mid_idx = n//2
            median = sorted_arr[mid_idx]
            return median
    else:
        #if even length -> add 2 middle /2 , odd length ->middle
        if(arr.ndim == 1):
            n = arr.size
            if(n%2==0): # even
                lower_idx = n//2 - 1 # "//" Integer Division to keep whole num and drop decimal 
                higher_idx = n//2
                median = (sorted_arr[lower_idx] + sorted_arr[higher_idx])/2
                return median
            else: # odd
                middle = n//2 
                median = np.take(sorted_arr, middle)
                return median
        else:
            if(axis==0):
                num_row = arr.shape[0]
                if(num_row%2==0): # even -> (sum 2 middle col or row) / 2
                    lower_row_idx = num_row//2-1
                    higher_row_idx = num_row//2
                    median = (sorted_arr[lower_row_idx] + sorted_arr[higher_row_idx])/2
                    return median
                else:
                    mid_row_idx = num_row//2
                    median = sorted_arr[mid_row_idx]
                    return median
            else:
                num_col = arr.shape[1]
                if(num_col%2==0): # even -> (sum 2 middle col or row) / 2
                    lower_col_idx = num_col//2-1
                    higher_col_idx = num_col//2
                    median = (sorted_arr[:,lower_col_idx] + sorted_arr[:,higher_col_idx])/2
                    return median
                else:
                    mid_col_idx = num_col//2
                    median = sorted_arr[:,mid_col_idx]
                    return median