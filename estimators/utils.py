import numpy as np

from scipy.special import expit


def irls(X, y, alpha, max_iter=1000):
    n, p = np.shape(X)
    beta = np.zeros((p,))
    pi = np.random.uniform(0, 1, size=(n,))
    H = X.T @ X + 4 * alpha * np.eye(p)
    invH = np.linalg.inv(H)
    i = 1
    stop = False
    while not (stop) and i <= max_iter:
        i += 1
        next_beta = beta + 4 * invH @ (X.T @ (y - pi) - alpha * beta)
        stop = np.allclose(beta, next_beta)
        logits = X @ next_beta
        pi = expit(logits)
        beta = next_beta
    return beta, pi
