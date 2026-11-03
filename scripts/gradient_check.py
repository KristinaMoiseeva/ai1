"""Centred finite-difference checks for analytic gradients.

API adapted from the course helper:
https://github.com/JustHawker/AIT_Course_SamU_2026/blob/main/lab_1-2/scripts/gradient_check.py
The assignment's three public entry points preserve their original arguments.
"""

from random import randrange

import numpy as np


def _check_point(x, h):
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.floating):
        raise TypeError("x must be a floating-point NumPy array")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """Estimate the gradient of scalar f(x), restoring x after every probe."""
    _check_point(x, h)
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        original = x[index].copy()
        try:
            x[index] = original + h
            positive = float(f(x))
            x[index] = original - h
            negative = float(f(x))
        finally:
            x[index] = original
        grad[index] = (positive - negative) / (2.0 * h)
        if verbose:
            print(index, grad[index])
    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """Estimate the vector-Jacobian product of array-valued f with df."""
    _check_point(x, h)
    df = np.asarray(df)
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        original = x[index].copy()
        try:
            x[index] = original + h
            positive = np.array(f(x), copy=True)
            x[index] = original - h
            negative = np.array(f(x), copy=True)
        finally:
            x[index] = original
        if positive.shape != df.shape or negative.shape != df.shape:
            raise ValueError("df must have the same shape as f(x)")
        grad[index] = np.sum((positive - negative) * df) / (2.0 * h)
    return grad


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """Print sampled numerical/analytic gradients; also return their records.

    Uses Python's random generator as the original helper does. Set
    random.seed(...) to make the sampled coordinates reproducible.
    """
    _check_point(x, h)
    analytic_grad = np.asarray(analytic_grad)
    if analytic_grad.shape != x.shape:
        raise ValueError("analytic_grad and x must have identical shapes")
    if x.size == 0:
        raise ValueError("x must not be empty")
    if not isinstance(num_checks, (int, np.integer)) or num_checks < 0:
        raise ValueError("num_checks must be a nonnegative integer")
    checks = []
    for _ in range(num_checks):
        index = tuple(randrange(size) for size in x.shape)
        original = x[index].copy()
        try:
            x[index] = original + h
            positive = float(f(x))
            x[index] = original - h
            negative = float(f(x))
        finally:
            x[index] = original
        numerical = (positive - negative) / (2.0 * h)
        analytic = float(analytic_grad[index])
        relative_error = abs(numerical - analytic) / max(1e-12, abs(numerical) + abs(analytic))
        print("numerical: %f analytic: %f, relative error: %e" % (numerical, analytic, relative_error))
        checks.append({
            "index": index, "numerical": numerical, "analytic": analytic,
            "relative_error": relative_error,
        })
    return checks


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """Compatibility helper for the optional blob-style API in the source."""
    inputs = tuple(inputs)
    gradients = []
    for blob in inputs:
        _check_point(blob.vals, h)
        gradient = np.zeros_like(blob.diffs)
        for index in np.ndindex(blob.vals.shape):
            original = blob.vals[index].copy()
            try:
                blob.vals[index] = original + h
                f(*(inputs + (output,)))
                positive = np.array(output.vals, copy=True)
                blob.vals[index] = original - h
                f(*(inputs + (output,)))
                negative = np.array(output.vals, copy=True)
            finally:
                blob.vals[index] = original
            gradient[index] = np.sum((positive - negative) * output.diffs) / (2.0 * h)
        gradients.append(gradient)
    return gradients


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    """Compatibility wrapper for objects exposing forward()."""
    return eval_numerical_gradient_blobs(lambda *args: net.forward(), inputs, output, h=h)
