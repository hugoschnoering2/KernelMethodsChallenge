from sklearn.base import BaseEstimator
from estimators.utils import irls
import numpy as np

"""

/!\ depreciated

class KernelLogRidgeEstimator(BaseEstimator):

    def __init__(self, kernel, alpha, **kwargs):
        super().__init__()
        self.kernel = kernel
        self.kwargs = kwargs
        self.X_fit = None
        self.weights = None
        self.alpha = alpha

    def fit(self, X, y, max_iter=1000):
        self.X_fit = X
        K = self._compute_kernel(X)
        self.weights = irls(K, y, self.alpha, max_iter=max_iter)[0]

    def predict(self, X):
        assert self.alpha is not None and self.X_fit is not None
        K = self._compute_kernel(X, self.X_fit)
        pred = K @ self.weights
        return np.sign(pred)

    def _compute_kernel(self, X1, X2=None):
        if self.kernel == "spectrum":
            from large_feature_space_kernels.spectrum_kernel import compute_spectrum_kernel
            compute_kernel = compute_spectrum_kernel
        K = compute_kernel(X1, X2, **self.kwargs)
        return K

"""
