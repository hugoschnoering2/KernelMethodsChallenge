import numpy as np

from estimators.estimatorABC import EstimatorABC
from estimators.utils import svm_solver


class SVM(EstimatorABC):
    def solve(self, K, y):
        return np.ravel(svm_solver(K, y, self.alpha))
