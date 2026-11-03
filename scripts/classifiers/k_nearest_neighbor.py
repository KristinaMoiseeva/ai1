"""kNN with Euclidean distance, implementing the course template's API.

Template and method signatures:
https://github.com/JustHawker/AIT_Course_SamU_2026/blob/main/lab_1-2/scripts/classifiers/k_nearest_neighbor.py
The three distance implementations and voting below are implemented in NumPy.
"""

import numpy as np


class KNearestNeighbor:
    """Memorize training samples and predict labels by k-neighbour voting."""

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """Store X[N,D] and integer labels y[N]; kNN has no fitted weights."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a nonempty two-dimensional array")
        if y.shape != (X.shape[0],) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain one integer label per training sample")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        self.X_train = X
        self.y_train = y.copy()

    def _validate_X(self, X):
        if self.X_train is None:
            raise RuntimeError("Call train before predicting or computing distances")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("X must have the same feature dimension as the training data")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        return X

    def predict(self, X, k=1, num_loops=0, batch_size=None):
        """Predict y[N] with the requested distance implementation.

        Optional batch_size bounds the temporary distance matrix to
        [batch_size, num_train], without changing neighbour selection.
        Equal distances keep the original training-row order; equal vote
        counts select the smallest label.
        """
        X = self._validate_X(X)
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(self.y_train):
            raise ValueError("k must be an integer between 1 and the training-set size")
        methods = {
            0: self.compute_distances_no_loops,
            1: self.compute_distances_one_loop,
            2: self.compute_distances_two_loops,
        }
        if num_loops not in methods:
            raise ValueError("num_loops must be 0, 1 or 2")
        if batch_size is not None and (
            not isinstance(batch_size, (int, np.integer)) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or None")
        if len(X) == 0:
            return np.empty(0, dtype=self.y_train.dtype)
        if batch_size is None:
            return self.predict_labels(methods[num_loops](X), k=k)
        prediction = np.empty(len(X), dtype=self.y_train.dtype)
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            prediction[start:stop] = self.predict_labels(methods[num_loops](X[start:stop]), k=k)
        return prediction

    def compute_distances_two_loops(self, X):
        """Compute L2 distances using one loop for each set of samples."""
        X = self._validate_X(X)
        dists = np.zeros((X.shape[0], self.X_train.shape[0]), dtype=np.float64)
        for i in range(X.shape[0]):
            for j in range(self.X_train.shape[0]):
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                difference = X[i] - self.X_train[j]
                dists[i, j] = np.sqrt(np.sum(difference * difference))
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """Compute L2 distances with one loop over the test samples."""
        X = self._validate_X(X)
        dists = np.zeros((X.shape[0], self.X_train.shape[0]), dtype=np.float64)
        for i in range(X.shape[0]):
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            difference = self.X_train - X[i]
            dists[i] = np.sqrt(np.sum(difference * difference, axis=1))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """Use ||x-t||² = ||x||² + ||t||² - 2*x·t, without sample loops."""
        X = self._validate_X(X)
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        test_squared = np.sum(X * X, axis=1, keepdims=True)
        train_squared = np.sum(self.X_train * self.X_train, axis=1)[None, :]
        squared_distances = test_squared + train_squared - 2.0 * X @ self.X_train.T
        # Floating-point cancellation can produce tiny negative squared distances.
        np.maximum(squared_distances, 0.0, out=squared_distances)
        return np.sqrt(squared_distances, out=squared_distances)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def predict_labels(self, dists, k=1):
        """Vote among k neighbours; ties are resolved by the smallest label."""
        if self.y_train is None:
            raise RuntimeError("Call train before predict_labels")
        dists = np.asarray(dists)
        if dists.ndim != 2 or dists.shape[1] != len(self.y_train):
            raise ValueError("dists must have shape [num_test, num_train]")
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(self.y_train):
            raise ValueError("k must be an integer between 1 and the training-set size")
        if not np.isfinite(dists).all() or np.any(dists < 0):
            raise ValueError("Distances must be finite and nonnegative")
        y_pred = np.empty(dists.shape[0], dtype=self.y_train.dtype)
        for i in range(dists.shape[0]):
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            indices = np.argsort(dists[i], kind="stable")[:k]
            closest_y = self.y_train[indices]
            labels, counts = np.unique(closest_y, return_counts=True)
            # np.unique sorts labels, and argmax chooses the first maximum.
            y_pred[i] = labels[np.argmax(counts)]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred
