"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from scipy.stats import multivariate_normal



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    post = np.zeros((n, K))  # responsibilities

    for k in range(K):
        # Construct full covariance matrix for component k (spherical)
        cov_k = mixture.var[k] * np.identity(d)
        # Evaluate pdf for each data point under component k
        post[:, k] = mixture.p[k] * multivariate_normal.pdf(X, mean=mixture.mu[k], cov=cov_k)

    # Normalize responsibilities across components
    total_density = post.sum(axis=1, keepdims=True)
    post /= total_density

    # Compute the log-likelihood
    log_likelihood = np.sum(np.log(total_density))

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    # Initialize updated parameters
    mu = np.zeros((K, d))
    var = np.zeros(K)
    p = np.zeros(K)

    # Compute N_k = sum of responsibilities for each component
    N_k = np.sum(post, axis=0)

    for k in range(K):
        # Update mean: weighted average
        mu[k] = np.sum(post[:, k][:, np.newaxis] * X, axis=0) / N_k[k]

        # Update variance: weighted average of squared distances from mean
        diff = X - mu[k]
        var[k] = np.sum(post[:, k] * np.sum(diff**2, axis=1)) / (d * N_k[k])

        # Update mixing proportion
        p[k] = N_k[k] / n

    return GaussianMixture(mu=mu, var=var, p=p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_likelihood = None
    max_iter = 100

    for _ in range(max_iter):
        # E-step
        post, log_likelihood = estep(X, mixture)

        # M-step
        mixture = mstep(X, post)

        # Convergence check with scaled condition
        if prev_likelihood is not None and abs(log_likelihood - prev_likelihood) <= 1e-6 * abs(log_likelihood):
            break
        prev_likelihood = log_likelihood

    return mixture, post, log_likelihood
