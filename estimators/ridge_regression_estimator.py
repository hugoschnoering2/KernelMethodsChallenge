import numpy as np

from estimators.estimatorABC import EstimatorABC


class RidgeRegression(EstimatorABC):
    def solve(self, K, y):
        return np.linalg.inv(K + K.shape[0] * self.alpha * np.eye(K.shape[0])) @ y
