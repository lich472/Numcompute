import numpy as np

class Node:
    def __init__(self, feature_index = None, threshold  = None, left = None, right = None, value = None):
        self.feature_index = feature_index  
        self.threshold = threshold          
        self.left = left                    # left subtree (=< threshold)
        self.right = right                  # right subtree (> threshold)
        self.value = value                  
    
class DecisionTreeClassifier:
    def __init__(self, max_depth = None, min_samples_split = 2, max_features = None, criterion = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.root = None
        self.fitted = False

        # lists used to accumulate historical chunks for streaming data
        self.X_all_chunks = []
        self.y_all_chunks = []

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        # save current batch into history for 'partial_fit' in case
        self.X_all_chunks = [X_arr]
        self.y_all_chunks = [y_arr]
        
        self.root = self.build_tree(X_arr, y_arr, depth=0)
        self.fitted = True
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_arr = np.asarray(X_chunk)
        y_arr = np.asarray(y_chunk)
        
        self.X_all_chunks.append(X_arr)
        self.y_all_chunks.append(y_arr)
        
        # rebuild tree on all data seen so far
        X_full = np.concatenate(self.X_all_chunks, axis=0)
        y_full = np.concatenate(self.y_all_chunks, axis=0)
        
        self.root = self.build_tree(X_full, y_full, depth=0)
        self.fitted = True
        return self

    def traverse(self, node, row):
        if node.value is not None:
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self.traverse(node.left, row)
        return self.traverse(node.right, row)

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        X_arr = np.asarray(X)
        return np.array([self.traverse(self.root, row) for row in X_arr])

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        
        # stop early if max depth, too few samples, or pure node
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (num_samples < self.min_samples_split) or \
           (num_classes == 1):
            return Node(value=self.leaf_value(y))
            
        split_idx, split_thresh = self.best_split(X, y)
        
        # no good split found
        if split_idx is None:
            return Node(value=self.leaf_value(y))
            
        # splitting masks
        left_mask = X[:, split_idx] <= split_thresh
        right_mask = X[:, split_idx] > split_thresh
        
        # recurse left and right
        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature_index=split_idx, threshold=split_thresh, left=left_child, right=right_child)

    def best_split(self, X, y):
        # scan across features and thresholds to find the best cut
        num_samples, num_features = X.shape
        if num_samples < self.min_samples_split:
            return None, None
            
        best_gain = -1.0
        best_idx = None
        best_thresh = None
        
        current_impurity = self.impurity(y)
        
        # random feature subset
        feature_indices = np.arange(num_features)
        if self.max_features is not None:
            if isinstance(self.max_features, int) and self.max_features < num_features:
                feature_indices = np.random.choice(num_features, self.max_features, replace=False)
                
        for idx in feature_indices:
            column_values = X[:, idx]
            unique_values = np.unique(column_values)
            
            # try each value as threshold
            for thresh in unique_values:
                left_mask = column_values <= thresh
                right_mask = column_values > thresh
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # skip if empty split
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                    
                # weighted impurity of children
                left_weight = len(y_left) / num_samples
                right_weight = len(y_right) / num_samples
                new_impurity = (left_weight * self.impurity(y_left)) + (right_weight * self.impurity(y_right))
                
                # compute Information Gain
                gain = current_impurity - new_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_thresh = thresh
                    
        return best_idx, best_thresh

    def impurity(self, y):
        if len(y) == 0:
            return 0.0
            
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        if self.criterion == 'gini':
            # gini = 1 - sum(probabilities^2)
            gini = 1 - np.sum(probabilities ** 2)
            return gini
        elif self.criterion == 'entropy':
            # entropy = -sum(probabilities * log2(probabilities))
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9)) # add epsilon to avoid log(0)
            return  entropy
        return 0.0

    def leaf_value(self, y):
        # return the majority class label present at this leaf node
        if len(y) == 0:
            return None
        classes, counts = np.unique(y, return_counts=True)
        best_index = np.argmax(counts)
        return classes[best_index]