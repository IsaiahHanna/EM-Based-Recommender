"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    log_posts = np.zeros((n, K))  # log posterior matrix

    for i in range(n):
        x_i = X[i]
        obs = x_i != 0  # observed indices

        for k in range(K):
            mu_k = mixture.mu[k][obs]
            var_k = mixture.var[k]
            p_k = mixture.p[k]

            # Compute the log of the Gaussian probability for observed dimensions
            diff = x_i[obs] - mu_k
            squared_dist = np.sum(diff ** 2)
            log_gauss = -0.5 * squared_dist / var_k
            log_gauss -= 0.5 * np.sum(obs) * np.log(2 * np.pi * var_k)

            # Add the log prior
            log_posts[i, k] = np.log(p_k) + log_gauss

    # Normalize using log-sum-exp to get posteriors
    log_likelihood = np.sum(logsumexp(log_posts, axis=1))
    posts = np.exp(log_posts - logsumexp(log_posts, axis=1, keepdims=True))

    return posts, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    new_mu = np.zeros((K, d))
    new_var = np.zeros(K)
    new_p = np.zeros(K)

    for k in range(K):
        post_k = post[:, k]  # shape (n,)
        weighted_sum = np.zeros(d)
        sum_weights = np.zeros(d)

        for i in range(n):
            mask = X[i] != 0  # observed entries
            weighted_sum[mask] += post_k[i] * X[i, mask]
            sum_weights[mask] += post_k[i]

        # Only update means where there is enough information
        mask_valid = sum_weights >= 1
        new_mu[k, mask_valid] = weighted_sum[mask_valid] / sum_weights[mask_valid]
        new_mu[k, ~mask_valid] = mixture.mu[k, ~mask_valid]  # keep previous value if not enough info

        # Compute variance
        num = 0
        denom = 0
        for i in range(n):
            mask = X[i] != 0
            diff = X[i, mask] - new_mu[k, mask]
            num += post_k[i] * np.sum(diff ** 2)
            denom += post_k[i] * np.sum(mask)

        new_var[k] = max(num / denom, min_variance) if denom > 0 else min_variance
        new_p[k] = np.sum(post_k) / n

    return GaussianMixture(new_mu, new_var, new_p)


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
    prev_ll = None
    threshold = 1e-6

    while True:
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)

        if prev_ll is not None and (ll - prev_ll) <= threshold * abs(ll):
            break
        prev_ll = ll

    return mixture, post, ll



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    p = mixture.p
    mu = mixture.mu
    var = mixture.var

    # Compute log probabilities of each x_i under each cluster j
    log_probs = np.zeros((n, K))

    for i in range(n):
        for j in range(K):
            mask = X[i] != 0  # mask for observed entries
            diff = X[i, mask] - mu[j, mask]
            sq_dist = np.sum(diff ** 2)
            log_prob = -0.5 * (np.sum(mask) * np.log(2 * np.pi * var[j]) + sq_dist / var[j])
            log_probs[i, j] = np.log(p[j]) + log_prob

    # Compute posterior probabilities (n, K)
    log_sums = logsumexp(log_probs, axis=1, keepdims=True)
    post = np.exp(log_probs - log_sums)

    # Fill missing entries with expected value under the posterior
    X_filled = X.copy()
    for i in range(n):
        for j in range(d):
            if X[i, j] == 0:
                X_filled[i, j] = np.dot(post[i], mu[:, j])

    return X_filled
