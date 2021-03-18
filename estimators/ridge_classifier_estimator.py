from estimators.estimatorABC import EstimatorABC
from estimators.utils import irls


class RidgeClassifier(EstimatorABC):
    def solve(self, K, y):
        return irls(K, y, self.alpha)[0]
