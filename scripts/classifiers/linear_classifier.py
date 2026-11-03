"""SGD-trained linear classifiers, compatible with the assignment's interface.

Template/API source:
https://github.com/JustHawker/AIT_Course_SamU_2026/blob/main/lab_1-2/scripts/classifiers/linear_classifier.py
Bias is supplied by the caller as the final constant feature. As in the
upstream template, regularization applies to every row of W, including bias.
"""

import numpy as np

from .softmax import softmax_loss_vectorized


class LinearClassifier:
    """Linear scores X@W, optimized by minibatch SGD with replacement."""

    def __init__(self, random_state=42):
        self.W = None
        self.rng = np.random.default_rng(random_state)

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """Update W and return one pre-update minibatch loss per SGD step."""
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a nonempty [N,D] array")
        if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0):
            raise ValueError("y must contain one nonnegative integer class per sample")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(reg) or reg < 0:
            raise ValueError("reg must be finite and nonnegative")
        if not isinstance(num_iters, (int, np.integer)) or num_iters < 0:
            raise ValueError("num_iters must be a nonnegative integer")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        num_train, dim = X.shape
        num_classes = int(np.max(y)) + 1
        if self.W is None:
            self.W = (0.001 * self.rng.standard_normal((dim, num_classes))).astype(X.dtype)
        elif self.W.ndim != 2 or self.W.shape[0] != dim or self.W.shape[1] < num_classes:
            raise ValueError("Existing W is incompatible with these features/classes")
        loss_history = []
        for it in range(num_iters):
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            indices = self.rng.choice(num_train, size=batch_size, replace=True)
            X_batch, y_batch = X[indices], y[indices]
            loss, grad = self.loss(X_batch, y_batch, reg)
            if not np.isfinite(loss) or not np.isfinite(grad).all():
                raise FloatingPointError("Non-finite loss/gradient; reduce the learning rate")
            loss_history.append(float(loss))
            self.W -= learning_rate * grad
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))
        return loss_history

    def predict(self, X):
        """Return the index of the largest linear score for each sample."""
        if self.W is None:
            raise RuntimeError("Call train before predict")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.W.shape[0]:
            raise ValueError("X has an incompatible feature dimension")
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return np.argmax(X @ self.W, axis=1)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def loss(self, X_batch, y_batch, reg):
        """Subclasses provide (mean loss, analytic gradient with respect to W)."""
        raise NotImplementedError("Use Softmax")



class Softmax(LinearClassifier):
    """Linear softmax classifier with cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
