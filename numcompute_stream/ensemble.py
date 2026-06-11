import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators = 10, max_depth = None, min_samples_split = 2, max_features = 'sqrt',criterion = 'gini'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        
        self.trees = []
        self.fitted = False
        
        self.X_all_chunks = []
        self.y_all_chunks = []
        

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def get_max_features(self, n_features):
        """
        Helper method to parse the max_features parameter.
        """
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features is None:
            return None
        return self.max_features
    
    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        self.X_all_chunks = [X_arr]
        self.y_all_chunks = [y_arr]
        
        n_features = X_arr.shape[1]
        tree_max_features = self.get_max_features(n_features)
        
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=tree_max_features,
                criterion=self.criterion
            )
            X_boot, y_boot = self.bootstrap_sample(X_arr, y_arr)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
        self.fitted = True
        return self
    
    def partial_fit(self, X_chunk, y_chunk):
        X_arr = np.asarray(X_chunk)
        y_arr = np.asarray(y_chunk)
        
        self.X_all_chunks.append(X_arr)
        self.y_all_chunks.append(y_arr)
        
        X_full = np.concatenate(self.X_all_chunks, axis=0)
        y_full = np.concatenate(self.y_all_chunks, axis=0)
        
        n_features = X_full.shape[1]
        tree_max_features = self.get_max_features(n_features)
        
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=tree_max_features,
                criterion=self.criterion
            )
            X_boot, y_boot = self.bootstrap_sample(X_full, y_full)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Call fit() first.")

        X_arr = np.asarray(X)

        # get predictions from all trees then majority vote
        all_predicts = np.array([tree.predict(X_arr) for tree in self.trees])

        final_predictions = []
        for i in range(X_arr.shape[0]):
            votes = all_predicts[:, i]
            classes, counts = np.unique(votes, return_counts=True)
            final_predictions.append(classes[np.argmax(counts)])

        return np.array(final_predictions)
        