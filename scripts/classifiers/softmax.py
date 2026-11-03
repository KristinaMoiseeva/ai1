"""Stable softmax cross-entropy and its analytic gradients.

Template/API source:
https://github.com/JustHawker/AIT_Course_SamU_2026/blob/main/lab_1-2/scripts/classifiers/softmax.py
Loss is computed as log-sum-exp minus the target logit; tiny target
probabilities are not clipped, so highly incorrect predictions keep their loss.
"""

import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """Return mean cross-entropy + reg*sum(W²), and dW, with explicit loops."""
    num_train, num_classes = X.shape[0], W.shape[1]
    if num_train == 0:
        raise ValueError("The minibatch must not be empty")
    loss = 0.0
    dW = np.zeros_like(W)
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        scores = X[i].dot(W)
        shifted = scores - np.max(scores)
        exponentials = np.exp(shifted)
        denominator = np.sum(exponentials)
        probabilities = exponentials / denominator
        loss += np.log(denominator) - shifted[y[i]]
        for j in range(num_classes):
            coefficient = probabilities[j] - (1.0 if j == y[i] else 0.0)
            dW[:, j] += coefficient * X[i]
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2.0 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return float(loss), dW


def softmax_loss_vectorized(W, X, y, reg):
    """Vectorized softmax cross-entropy with the same regularization convention."""
    num_train = X.shape[0]
    if num_train == 0:
        raise ValueError("The minibatch must not be empty")
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X @ W
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=1, keepdims=True)
    rows = np.arange(num_train)
    loss = np.mean(np.log(denominator[:, 0]) - shifted[rows, y])
    loss += reg * np.sum(W * W)
    score_gradient = exponentials / denominator
    score_gradient[rows, y] -= 1.0
    dW = X.T @ score_gradient / num_train + 2.0 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return float(loss), dW
