import numpy as np


class SubstringKernel():
    def __init__(self, lambd, k):
        self.lambd = lambd
        self.b_arr = {}
        self.k_arr = {}
        self.k = k

    def b(self, k, x1, x2):
        if k == 0:
            return 1
        if len(x1) < k or len(x2) < k:
            return 0
        if (k, x1, x2) not in self.b_arr:
            self.b_arr[(k, x1, x2)] = self.lambd * self.b(k, x1[:-1], x2) + \
                                      self.lambd * self.b(k, x1, x2[:-1]) - \
                                      self.lambd ** 2 * self.b(k, x1[:-1], x2[:-1]) + \
                                      int(x1[-1] == x2[-1]) * self.lambd ** 2 * self.b(k - 1, x1[:-1], x2[:-1])
        return self.b_arr[(k, x1, x2)]

    def K(self, k, x1, x2):
        if k == 0:
            return 1
        if len(x1) < k or len(x2) < k:
            return 0
        if (k, x1, x2) not in self.k_arr:
            self.k_arr[(k, x1, x2)] = self.K(k, x1[:-1], x2) + \
                                      self.lambd ** 2 * np.sum([self.b(k - 1, x1[:-1], x2[:j - 1]) if x2[j] == x1[-1] else 0
                                                                for j in range(len(x2[:-1]))]
                                                               )
        return self.k_arr[(k, x1, x2)]

    def compute_substring_kernel(self, X1, X2=None):
        if X2 is None:
            K = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                print(len(self.k_arr))
                for j in range(i, len(X1)):
                   kij = self.K(self.k, X1[i], X1[j])
                   K[i,j] = kij
                   K[j,i] = kij
            return K
        K = np.zeros((len(X1), len(X2)))
        for j in range(len(X1)):
            for i in range(len(X2)):
                kij = self.K(self.k, X1[j], X2[i])
                K[i, j] = kij
        return K