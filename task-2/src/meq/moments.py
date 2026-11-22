"""Exact permutation moments for the text and style statistics.

These compute the mean and variance over every fixed-size relabeling, without
enumeration. They do not assert normal tails or turn non-rejection into equality.
The underlying kernel, style features, and observed statistics are unchanged.
"""
from functools import lru_cache
from itertools import product
import math

import numpy as np


def falling(n, k):
    return math.prod(range(n-k+1, n+1)) if 0 <= k <= n else 0


def nonnegative(value, magnitude):
    """Clip rounding error, but expose a materially incorrect variance."""
    tolerance = 256 * np.finfo(float).eps * max(magnitude, 1e-300)
    if value < -tolerance:
        raise ArithmeticError(f'Negative exact variance: {value}, scale {magnitude}')
    return max(0.0, float(value))


@lru_cache(maxsize=None)
def edge_moments(a, b):
    """Products of coefficients for equal, adjacent, and disjoint edges."""
    n = a+b
    def coefficient(x, y):
        if x == y == 1:
            return 2 / (a*(a-1))
        if x == y == 0:
            return 2 / (b*(b-1))
        return -2 / (a*b)
    def expectation(size, edges):
        total = 0.
        for assignment in product((0, 1), repeat=size):
            k = sum(assignment)
            probability = falling(a, k)*falling(b, size-k)/falling(n, size)
            total += probability*math.prod(coefficient(assignment[i], assignment[j]) for i, j in edges)
        return total
    return (expectation(2, [(0, 1), (0, 1)]),
            expectation(3, [(0, 1), (0, 2)]),
            expectation(4, [(0, 1), (2, 3)]))


def mmd_moments(matrix, a):
    """Exact null mean (zero) and variance for a symmetric kernel matrix."""
    k = np.asarray(matrix, dtype=float).copy()
    n = len(k); b = n-a
    if min(a, b) < 2:
        raise ValueError('MMD needs at least two samples in each group')
    np.fill_diagonal(k, 0.)
    # Adding a constant to all off-diagonal entries leaves every MMD unchanged.
    off_diagonal_mean = k.sum()/(n*(n-1))
    k -= off_diagonal_mean
    np.fill_diagonal(k, 0.)
    edge_square = float(np.sum(k*k)/2)
    row_square = np.sum(k*k, axis=1)
    adjacent = float(np.sum(np.square(k.sum(axis=1))-row_square)/2)
    edge_sum = float(k.sum()/2)
    disjoint = (edge_sum*edge_sum-edge_square)/2-adjacent
    equal_weight, adjacent_weight, disjoint_weight = edge_moments(a, b)
    terms = [edge_square*equal_weight, 2*adjacent*adjacent_weight, 2*disjoint*disjoint_weight]
    return 0., nonnegative(sum(terms), sum(map(abs, terms)))


def mmd(a, b, permutations=300, seed=0):
    """Exact null normalization; seed and permutations retained for API compatibility."""
    from . import features
    observed = variance = 0.; count = 0
    for pid in sorted(a.keys() & b.keys()):
        xs, ys = a[pid], b[pid]
        na, nb = len(xs), len(ys)
        if min(na, nb) < 2:
            continue
        k = features.kernel_matrix(xs+ys)
        observed += (k[:na, :na].sum()/(na*(na-1))
                     + k[na:, na:].sum()/(nb*(nb-1))
                     - 2*k[:na, na:].sum()/(na*nb))
        variance += mmd_moments(k, na)[1]
        count += 1
    return {'prompts': count, 'stat': float(observed),
            'z': float(observed/math.sqrt(variance)) if variance else 0.}


def style_moments(rows, a):
    """Mean/variance of ||mean(A)-mean(B)||² and covariance of its vector."""
    rows = np.asarray(rows, dtype=float)
    n = len(rows); b = n-a
    if min(a, b) < 1:
        raise ValueError('Style comparison needs nonempty groups')
    x = rows-rows.mean(axis=0)
    gram = x.T @ x
    d1 = float(np.trace(gram))
    d2 = float(np.square(np.sum(x*x, axis=1)).sum())
    k2 = float(np.square(gram).sum())
    p1, p2, p3, p4 = [falling(a, r)/falling(n, r) if r <= n else 0. for r in range(1, 5)]
    q_mean = (p1-p2)*d1
    terms = [(p1-7*p2+12*p3-6*p4)*d2,
             (p2-2*p3+p4)*(d1*d1+2*k2), -q_mean*q_mean]
    factor = n/(a*b)
    mean = factor**2*q_mean
    variance = factor**4*nonnegative(sum(terms), sum(map(abs, terms)))
    covariance = n/(a*b*(n-1))*gram
    return mean, variance, covariance


def style_null(blocks):
    """Moments of the squared norm of the average of independent block shifts."""
    count = len(blocks)
    if not count:
        return 0., 0.
    mean = variance = 0.
    covariance_sum = np.zeros((blocks[0][0].shape[1],)*2)
    for rows, a in blocks:
        m, v, covariance = style_moments(rows, a)
        mean += m
        variance += v+4*float(np.sum(covariance_sum*covariance))
        covariance_sum += covariance
    return mean/count**2, nonnegative(variance, abs(variance))/count**4


def style_test(a, b, scale, permutations=300, seed=0):
    """Same observed style shift, exact relabeling mean and standard deviation."""
    mean, spread = scale
    blocks = []
    for pid in sorted(a.keys() & b.keys()):
        xs, ys = np.asarray(a[pid], dtype=float), np.asarray(b[pid], dtype=float)
        if not len(xs) or not len(ys):
            continue
        blocks.append(((np.vstack([xs, ys])-mean)/spread, len(xs)))
    if not blocks:
        return {'prompts': 0, 'z': 0., 'shift': 0.}
    shift = sum((rows[:a].mean(axis=0)-rows[a:].mean(axis=0) for rows, a in blocks)) / len(blocks)
    observed = float(np.sum(shift*shift))
    null_mean, variance = style_null(blocks)
    return {'prompts': len(blocks), 'shift': observed,
            'z': (observed-null_mean)/math.sqrt(variance) if variance else 0.}
