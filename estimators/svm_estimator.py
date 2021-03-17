import numpy as np
from sklearn.base import BaseEstimator

from estimators.utils import svm_solver


class SVM(BaseEstimator):

    def __init__(self, kernel="mismatch", k=3, m=0, alpha=0, trie=None):
        super().__init__()

        self.kernel = kernel
        self.alpha = alpha

        # args for spectrum / mismatch
        self.m = m
        self.k = k
        self.trie = trie
        self.lookup_table = None

        self.coef = None

    def fit(self, X, y):

        # computing the kernel
        if self.kernel in ["spectrum", "mismatch"]:
            from large_feature_space_kernels.mismatchingTrie import MismatchingTrie
            self.trie = MismatchingTrie(m=self.m, n=len(X)).add_data(X, k=self.k)
            self.trie.compute_kernel_matrix()
            K = self.trie.K
        else:
            raise NotImplementedError

        # solving the dual formulation of the SVM problem
        self.coef = np.ravel(svm_solver(K, y, self.alpha))

        if self.kernel in ["spectrum", "mismatch"]:
            self.lookup_table = self.trie.build_lookup_table(self.coef)

    def predict(self, X):
        if self.kernel in ["spectrum", "mismatch"]:
            y_pred = np.array([self.predict_mismatch(sample) for sample in X])
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
