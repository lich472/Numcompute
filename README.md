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
