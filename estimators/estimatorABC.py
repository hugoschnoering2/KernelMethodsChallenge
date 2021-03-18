from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from large_feature_space_kernels.mismatchingTrie import MismatchingTrie
from large_feature_space_kernels.substring import SubstringKernel


class EstimatorABC(BaseEstimator):
    def __init__(self, kernel="mismatch", k=3, m=0, alpha=0, trie=None, lambd=1):
        super().__init__()

        self.kernel = kernel
        self.alpha = alpha

        # args for spectrum / mismatch
        self.m = m
        self.k = k
        self.trie = trie
        self.lookup_table = None

        # substring kernel
        self.substring_kernel = None
        self.lambd = lambd
        self.X_fit = None

        self.coef = None

    @abstractmethod
    def solve(self, K, y):
        pass

    def fit(self, X, y, K=None):

        # computing the kernel
        if self.kernel == "mismatch":
            if K is None:
                self.trie = MismatchingTrie(m=self.m, n=len(X)).add_data(X, k=self.k)
                self.trie.compute_kernel_matrix()
                K = self.trie.K
        elif self.kernel == "substring":
            self.substring_kernel = SubstringKernel(self.k, self.lambd)
            K = self.substring_kernel.compute_substring_kernel(X)
            self.X_fit = X
        else:
            raise NotImplementedError

        self.coef = self.solve(K, y)

        if self.kernel == "mismatch":
            self.lookup_table = self.trie.build_lookup_table(self.coef)

    def predict(self, X):
        if self.kernel == "mismatch":
            y_pred = np.array([self.predict_mismatch(sample) for sample in X])
        elif self.kernel == "substring":
            y_pred = self.substring_kernel.compute_substring_kernel(self.X_fit, X)
        else:
            raise NotImplementedError
        return np.sign(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y) * 100.

    def predict_mismatch(self, sample):
        res = 0
        for i in range(len(sample) - self.k):
            try:
                res += self.lookup_table[sample[i:i + self.k]]
            except:
                pass
        return res
