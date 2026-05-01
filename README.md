# Numcompute

### `sort_search.py`
- **Sorting**:
  - Stable sort wrapper (`np.sort(kind='stable')`): sort array and keep original position if value is equal
    - input: str or num ndim array --> output: a sorted array
    - Sample: input: arr = [[1,2,3,5,4,5],["Bob","Alice","Henry","Alice","Luca","Braydon"]]
 -> expected output: [['1' '2' '3' '4' '5' '5'] ['Alice' 'Alice' 'Bob' 'Braydon' 'Henry' 'Luca']]

  - Multi-key sort (sort by multiple columns): sort array followed by the specific the order of column (keys)
    - input (data, order of sorted column respectively, ascending = True(by default)) supporting str or num array --> output: sorted array 
    - Sample: 
        - Input: 
            arr = np.array([["Alice", 6, 2], 
                ["Bob", 2, 5], 
                ["Alice", 3, 1]]) 

            arr = np.array([["Alice", 6, 2], 
                ["Bob", 2, 5], 
                ["Alice", 3, 1]])

                print(multi_key_sort(arr,[0,1], False)) 
                print(multi_key_sort(arr,[0,1], True))
        - Output: 

                [['Bob' '2' '5']
                ['Alice' '6' '2']
                ['Alice' '3' '1']]
                
                [['Alice' '3' '1']
                ['Alice' '6' '2']
                ['Bob' '2' '5']]
- **Top-k / Partial Sort**:
  - `topk(values, k, largest=True, return_indices=True)` using `np.argpartition` : return the top k number or indices depending on largest(largest = True -> top biggest num, False -> top smallest num)
    - Input: num array associated with k (top number of the list array) with largest = True and return_indices=True by default -> top k UNSORTED number 
    - Sample: 
        - Input: 

            Array [0.95, 0.87, 0.99, 0.77, 0.91], k =3
                print(topk(arr, 3, True, False))
                print(topk(arr, 3, False, False))
                print(topk(arr, 3, False, True))
                print(topk(arr, 3, True, True))

        - Output: 

                [0.91 0.95 0.99]
                [0.77 0.91 0.87]
                [3 4 1]
                [4 0 2]

  - Implement **quickselect** for educational purposes: create a helper function partition() # using Lomuto's Partitioning (with choosing pivot is the last number of arr list)
    - partition() method to find new pivot position with new partition array (the value from the left side of pivot is less or equal than pivot and the right side of pivot is larger or equal than pivot)
    - quickselect() from the new partition arr list associated with pivot position we easily search the expected kth smallest number 
    - Sample Input: 

        scores = [1, 3, 10, 7, 5, 8, 4]
        print(quickselect(scores,5))

    - Output: 

        7 # from partition arr [ 1  3  4  5  7  8 10] 
- **Searching**:
  - `binary_search(sorted_array, x)` returning insertion index and existence boolean
    - Input: 

        arr = [1, 3, 5, 7, 9, 11]
        print(binary_search(arr, -1))
        print(binary_search(arr, 8))
        print(binary_search(arr, 9))
        print(binary_search(arr, 12))

    - Output:

        (0, False)
        (4, False)
        (4, True)
        (6, False)

### `stats.py`
- Basic descriptive statistics:
  - Mean, median, standard deviation, min, max
    - mean() method - compute the mean value of data based on axis with shape (m,n)
        - axis = None --> compute all value --> return scalar
        - axis = 0 --> compute per column --> return mean array with shape (n,)
        - axis = 1 --> compute per row --> return mean array with shape (m,)
        - Sample: 

            - Input: 
                    arr = np.array([[1,2,3],
                    [4,5,6]])

                    print(mean(arr))
                    print(mean(arr, axis=0))
                    print(mean(arr, axis=1))

            - Output:
                    3.5
                    (array([2.5, 3.5, 4.5]), (3,))
                    (array([2., 5.]), (2,))
    
    - median() method - middle value, sort everything, find the center
        - Need to consider the length of number arr --> if even length -> add 2 middle /2 , odd length -> middle
        - Work along with axis 
            - axis = 0 -> find center of row with its shape
            - axis = 1 -> find center of column with it shape
            - axis = None -> median of all value -> return scalar 
        - Sample:
            - Input:
                    arr = np.array([[1,2,3,4],
                    [4,5,6,7],
                    [7,8,9,10]])

                    print(median(arr, axis=0))
                    print(median(arr, axis=1))
                    print(median(arr))
            
            - Output:
                    print(median(arr, axis=0))
                    print(median(arr, axis=1))
                    print(median(arr))

    - std() method to compute standard deviation with mean() and keepdims here to preserve the collapsed dimension as size 1, making broadcasting work correctly in all cases. 
        - The formula: std = sqrt( sum( (x - mean)² ) / n ) 
        - Integrate with ddof for both Popular (ddof = 0) and Sample (ddof = 1) standard deviation
            - Sample: 
            - Input:
                    arr = np.array([[1,2,3],
                    [4,5,6]])

                    print(std(arr,axis=None))
                    print(std(arr,axis=0))
                    print(std(arr,axis=1))
                    print(std(arr,axis=None, ddof=1))
            
            - Output:
                    1.707825127659933
                    [1.5 1.5 1.5]
                    [0.81649658 0.81649658]
                    1.8708286933869707
    - min() return the smallest value based on axis 
        - Axis = 0 -> return the smallest value per column 
        - Axis = 1 -> return the smallest value per row 
        - Axis = None -> return the smallest value of all value
            - Sample: 
            - Input: 
                    arr = np.array([[1,2,3],
                                    [4,5,6]])
                    print(min(arr, axis=0))
                    print(min(arr, axis=1))
                    print(min(arr, axis=None))

            - Output with its shape:
                    (array([1, 2, 3]), (3,))
                    (array([1, 4]), (2,))
                    (1, ())
    - max() return the largest of value based on axis
        - Axis = 0 -> return the largest value per column
        - Axis = 1 -> return the largest value per row
        - Axis = None -> return the largest value of all value
            - Sample:
            - Input: 
                    arr = np.array([[1,2,3],
                                    [4,5,6]])
                    print(max(arr, axis=0))
                    print(max(arr, axis=1))
                    print(max(arr, axis=None))

            - Output:
                    (array([4, 5, 6]), (3,))
                    (array([3, 6]), (2,))
                    (6, ())
- Histogram - Divides a range of values into bins and counts how many elements fall into each bin
    - Input: data - numeric data, bins - number of equal-width bins to divide the range into
    - Output: edges, counts
    - Sample:
        - Input:
                arr = np.array([[1,2,3,4],
                                [5,6,7,8],
                                [9,10,11,12]])

                print(histogram(arr, bins=5))
        - Output
                (array([ 1. ,  3.2,  5.4,  7.6,  9.8, 12. ]), array([3, 2, 2, 2, 3]))
            - Meanings:
                - bins 0: [1, 3.2) -> count 3 #[1,2,3]
                - bins 1: [3.2, 5.4) -> count 2 #[4,5]
                - bins 2: [5.4, 7.6) -> count 2 #[6,7]
                - bins 3: [7.6, 9.8) -> count 2 #[8,9]
                - bins 4: [9.8, 12] -> count 3 #[10,11,12]
- Quantiles (with NaN handling) - same concept with Percentile by different q scale (with Quantitles q from 0->1 and Percentile q from 0->100)
    - Input: data, q scale from 0->1 and interpolation="linear" by default - the most mathematically standard interpolation for quantiles
    - Output: return the value based on q, for example: 0.5 is 50%, 0.75 is 75%, ...
    - Handle with NaN value by drop all NaN that value as filling with mean changes the distribution of your data — it affects where quantiles land
    - Sample:
        - Input: data = [10, 20, np.nan, 40, 50]
            - print(quantiles(data,0))
            - print(quantiles(data,0.5))
            - print(quantiles(data,0.75))
            - print(quantiles(data,1))
        - Output: 
            - 10.0
            - 30.0
            - 42.5
            - 50.0

### `rank.py`
- `rank(data, method='average'|'dense'|'ordinal')` — handle ties (occuring the duplicate value)
    - dense rank -> 0 is the lowest and ties the the same rank
    - ordinal rank -> 0 is the lowest, ties is still treated as a unique value 
    - average rank -> the average of the ranks they would have occupied
    - Sample:
        - Input:
            - scores_with_ties = np.array([0.7, 0.9, 0.8, 0.8, 0.8])
            - print(rank(scores_with_ties, 'average'))
            - print(rank(scores_with_ties, 'ordinal'))
            - print(rank(scores_with_ties, 'dense'))
        - Output:
            - [0. 4. 2. 2. 2.]
            - [0 4 1 2 3]
            - [0 2 1 1 1]
- `percentile(data, q, interpolation='linear'|'lower'|'higher'|'midpoint')`
    - `the formula: `
    - Example with `score = np.array([10, 20, 30, 40, 50])`

            The Position Formula
            position = (q / 100) * (n - 1)

            q=0   → position 0.0  → index 0  → value 10
            q=50  → position 2.0  → index 2  → value 30
            q=100 → position 4.0  → index 4  → value 50
            q=30  → position 1.2  → between index 1 and 2

            The Four Interpolation Answers To Position 1.2
            lower    → take index 1         → 20
            higher   → take index 2         → 30
            midpoint → average both         → 25
            linear   → 20 + 0.2*(30-20)    → 22

    - Input: data - numerics array, q scale from 1->100 but could create a different scale array to save time and interpolation
        - `lower` choosing the lower position of array value 
        - `higher` choosing the higer position of array value 
        - `lower` and `linear` is efficient when position of value is odd
        - `average` mean of value from higher_idx and lower_idx
        - `linear` how far between the two values you actually are

    - Output: value of array based on q scale and its interpolation

    - Sample:
        - Input:

                score = np.array([10, 20, 30, 40, 50])
                print(percentile(score, np.array([50,30]), 'linear'))
                print(percentile(score, 30, 'average')) 
                print(percentile(score, 100, 'linear')) 

        - Output:

                [30. 22.]
                25.0
                50.0 