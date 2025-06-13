import numpy as np
from math import ceil, log
from tqdm import tqdm

class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, contamination=0.01):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.trees = [] # trees in the forest

    # recursive tree building
    def build_tree(self, X, current_depth=0, max_depth=8):
        if current_depth >= max_depth or len(X) <= 1:
            return {'type': 'leaf', 'size': len(X)}
        
        random_feature_idx = np.random.randint(0, X.shape[1])
        threshold = np.random.choice(X[:, random_feature_idx])

        left_subtree = X[X[:, random_feature_idx] <= threshold]
        right_subtree = X[X[:, random_feature_idx] > threshold]
        
        return {
            'type': 'node',
            'feature': random_feature_idx,
            'threshold': threshold,
            'left_subtree': self.build_tree(left_subtree, current_depth+1, max_depth),
            'right_subtree': self.build_tree(right_subtree, current_depth+1, max_depth)
        }

    # go through the tree and count the path to the sample
    def _count_way_length(self, sample, tree, depth=0):
        if tree['type'] == 'leaf':
            return depth + 2 * (log(tree['size'] + 0.5) if tree['size'] > 1 else 1)
        if sample[tree['feature']] <= tree['threshold']:
            return self._count_way_length(sample, tree['left_subtree'], depth+1)
        else:
            return self._count_way_length(sample, tree['right_subtree'], depth+1)

    def fit(self, X):
        X = np.array(X)
        max_depth = ceil(log(self.max_samples, 2))
        
        for _ in tqdm(range(self.n_estimators), desc="Seaching for anomalies"):
            sample_idxs = np.random.choice(X.shape[0], size=min(self.max_samples, X.shape[0]), replace=False)
            X_sample = X[sample_idxs]
            tree = self.build_tree(X_sample, max_depth=max_depth)
            self.trees.append(tree)

        return self

    def predict(self, X) -> np.array: # returns (true/false) array
        X = np.array(X)
        anomaly_scores = []
        
        # count path lengths and anomaly scores for each sample
        for x in X:
            path_lengths = []
            for tree in self.trees:
                path_lengths.append(self._count_way_length(x, tree))
            avg_path = np.mean(path_lengths)
            anomaly_score = 2 ** (-avg_path / (2 * log(len(X) - 1) + 0.5772156649 if len(X) > 1 else 1))
            anomaly_scores.append(anomaly_score)
        
        threshold = np.quantile(anomaly_scores, 1 - self.contamination)
        return np.array(anomaly_scores) >= threshold