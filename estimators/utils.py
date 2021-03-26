import cvxopt
import numpy as np
from cvxopt import matrix, solvers
from scipy.special import expit

cvxopt.solvers.options['show_progress'] = False


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


def svm_solver(K, y, alpha):
    n = K.shape[0]
    q = matrix(-y, tc='d')
    P = matrix(K, tc='d')
    G = []
    h = []
    for i in range(n):
        c1 = np.zeros(n)
        c1[i] = y[i]
        G.append(c1)
        h.append(1 / (2 * alpha * n))
        c2 = np.zeros(n)
        c2[i] = -y[i]
        G.append(c2)
        h.append(0)
    G = matrix(np.array(G), tc='d')
    h = matrix(np.array(h), tc='d')
    sol = solvers.qp(P, q, G, h)['x']
    return sol

def ensemble_predictions(list_of_predictions, label_of_zero=0):
    """
    list_of_predictions is a list of array of 0 and 1, not -1 and 1 !!!
    """
    assert len(list_of_predictions) % 2 == 1
    final_predictions = np.sum(np.array(list_of_predictions), axis=0)
    if label_of_zero == 0:
        final_predictions = final_predictions > len(list_of_predictions) / 2
    elif label_of_zero == -1:
        final_predictions = np.sign(final_predictions)
    else:
        raise ValueError
    return final_predictions.astype(np.int)
