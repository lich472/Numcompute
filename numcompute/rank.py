def rank(data, method):
    match(method.lower()): 
        case 'dense': #Dense ranks (0 = lowest)
            sorted_unique = np.unique(data)
            rank_map = {val: i for i, val in enumerate(sorted_unique)}
            dense_ranks = np.vectorize(rank_map.get)(data) #np.vectorize() which takes a nested sequence of objects or numpy arrays as inputs and returns a single numpy array or a tuple of numpy arrays
            return dense_ranks
        case 'ordinal':
            ordinal_rank = np.argsort(np.argsort(data, kind="stable")) # np.argsort() return the opsition od sorted arr, np.argsort(np.argsort()) return the rank( 0-> smallest)
            return ordinal_rank
        case 'average':
            #scores_with_ties = np.array([0.7, 0.9, 0.8, 0.8])
            ordinal_rank = np.argsort(np.argsort(data, kind="stable")) # 0 3 1 2
            sorted_unique_count = np.unique(data, return_counts=True, return_inverse=True)
            sorted_arr_inverse = sorted_unique_count[1] # 0 2 1 1 == dense_rank
            sorted_arr_count= sorted_unique_count[2] # 1 2 1 - numer of duplicate element
            #map each data element associated with sorted_inverse
            result= np.bincount(sorted_arr_inverse, weights=ordinal_rank) # group0(0.7)=0, group1(0.8)=1+2=3, group2(0.9)=3 -> 0 3 3 
            per_group_average = result / sorted_arr_count    # mean per unique group -> 0 1.5 3 (0/1, 3/2, 3/1)
            per_element_average = per_group_average[sorted_arr_inverse]  # map back to original positions
            print(sorted_unique_count)
            return per_element_average

# scores_with_ties = np.array([0.7, 0.9, 0.8, 0.8, 0.8])
# print(rank(scores_with_ties, 'average'))
            
### percentile()

# The Position Formula
# position = (q / 100) * (n - 1)

# q=0   → position 0.0  → index 0  → value 10
# q=50  → position 2.0  → index 2  → value 30
# q=100 → position 4.0  → index 4  → value 50
# q=30  → position 1.2  → between index 1 and 2

# The Four Interpolation Answers To Position 1.2
# lower    → take index 1         → 20
# higher   → take index 2         → 30
# midpoint → average both         → 25
# linear   → 20 + 0.2*(30-20)    → 22
# Linear uses the fractional part 0.2 as a weight — how far between the two values you actually are.

def percentile(data, q: np.ndarray, interpolation):
    q_array = np.array(q)
    if(np.any(q_array<0)or np.any(q_array>100)):
        raise ValueError("q should be from 0 to 100")
    if( not np.issubdtype(data.dtype, np.number) ):
        raise ValueError("Only numerics array is allowed.")
    if( data.ndim != 1):
        raise ValueError("Only 1D array is allowed.")
    sorted_arr = np.sort(data, kind='stable')
    position = (q_array/100) * (len(data)-1)
    lower_idx = np.floor(position).astype(int)
    higher_idx = np.clip(lower_idx + 1, 0, len(data)-1) # using np.clip() here to prevent out of bounds
    match(interpolation.lower()):
        #'linear'|'lower'|'higher'|'midpoint'
        case 'lower': 
            return sorted_arr[lower_idx]
        case 'higher':
            # if(position % 1 == 0): # if integer/whole number -> take lower_idx
            #     return sorted_arr[lower_idx]
            # else:
            #     return sorted_arr[higher_idx]
            val = np.where(position % 1 == 0, sorted_arr[lower_idx], sorted_arr[higher_idx])
            return val
        case 'average':
            # val = (sorted_arr[head_position] + sorted_arr[redundant_position])/2
            # if(position % 1 == 0): # if integer/whole number -> take lower_idx
            #     return sorted_arr[lower_idx]
            # else:
            #     val = (sorted_arr[lower_idx] + sorted_arr[higher_idx])/2 # using int() to return the lowest whole numer and round() to return the highest number
            #     return val
            val = np.where(position % 1 == 0, sorted_arr[lower_idx], (sorted_arr[lower_idx] + sorted_arr[higher_idx])/2)
            return val
        case 'linear':
            # fomular: sorted_arr[head_position] + fraction*(sorted_arr[redundant_position] - sorted_arr[head_position])
            fraction = position - lower_idx
            val = sorted_arr[lower_idx] + fraction*(sorted_arr[higher_idx] - sorted_arr[lower_idx])
            return val
score = np.array([10, 20, 30, 40, 50])
print(score.dtype)
print(percentile(score, np.array([50,30]), 'linear'))
print(percentile(score, 50, 'average')) 
print(percentile(score, 100, 'linear'))        
