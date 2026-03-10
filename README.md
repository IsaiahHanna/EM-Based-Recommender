# EM-Based Recommender System

This project implements a **collaborative filtering recommender system using a Mixture of Gaussians trained with the Expectation–Maximization (EM) algorithm**. The model learns latent user preference clusters from a **sparse movie ratings matrix** and predicts missing ratings using probabilistic inference.

The implementation is written from scratch using **NumPy and SciPy**, emphasizing clarity and direct implementation of the EM algorithm.

This project was completed as part of the **MITx 6.86x: Machine Learning with Python – From Linear Models to Deep Learning** course.

---

# Overview

Recommender systems often deal with **sparse matrices** where users rate only a small subset of items. This project models user rating behavior using a **Gaussian mixture model**, where each component represents a latent user type.

The system:

1. Learns latent user clusters using **EM with incomplete data**
2. Handles missing ratings directly during training
3. Predicts missing ratings using **expected values from the learned mixture model**
4. Evaluates model quality using **log-likelihood, BIC, and RMSE**

---

# Key Concepts

This project demonstrates several important machine learning concepts:

- Expectation–Maximization (EM)
- Mixture of Gaussians
- Soft clustering
- Matrix completion
- Probabilistic recommender systems
- Handling missing data during training

The EM algorithm alternates between:

### E-Step
Compute the **posterior probability** that each user belongs to each latent cluster.

### M-Step
Update mixture parameters (means, variances, and mixture weights) to maximize likelihood.

Iterations continue until the **log-likelihood converges**.

---
