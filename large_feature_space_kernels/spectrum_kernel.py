
import numpy as np

from numba import njit
from large_feature_space_kernels.trie import Trie

@njit
def spectrum_kernel_on_sorted_keys(keys_a, keys_b):
    ia, ib = 0, 0
    res = 0
    while ia < keys_a.shape[0] and ib < keys_b.shape[0]:
        if keys_a[ia] < keys_b[ib]:
            ia += 1
        elif keys_a[ia] > keys_b[ib]:
            ib += 1
        else:
            common_key = keys_a[ia]
            count_a = 1
            count_b = 1
            while (ia+1 < len(keys_a)) and (keys_a[ia+1] == common_key):
                count_a += 1
                ia += 1
            while (ib+1 < len(keys_b)) and (keys_b[ib+1] == common_key):
                count_b += 1
                ib += 1
            res += count_a * count_b
            ia += 1
            ib += 1
    return res

def compute_spectrum_kernel(X1, X2=None, k=1):
    if X2 is None:
            keys = []
            for i in range(len(X1)):
                s = X1[i]
                t = Trie()
                for i in range(0, len(s)-k):
                    t.add_word(s[i:i+k])
                keys.append(t.printasc())
            keys = np.array(keys)
            K = np.zeros(shape=(len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(i, len(X1)):
                    K[i, j] = spectrum_kernel_on_sorted_keys(keys[i], keys[j])
            K = (K + K.T) / 2
    else:
            keys1 = []
            keys2 = []
            for i in range(len(X1)):
                s = X1[i]
                t = Trie()
                for i in range(0, len(s)-k):
                    t.add_word(s[i:i+k])
                keys1.append(t.printasc())
            keys1 = np.array(keys1)
            for i in range(len(X2)):
                s = X2[i]
                t = Trie()
                for i in range(0, len(s)-k):
                    t.add_word(s[i:i+k])
                keys2.append(t.printasc())
            keys2 = np.array(keys2)
            K = np.zeros(shape=(len(X1), len(X2)))
            for i in range(0, len(X1)):
                for j in range(0, len(X2)):
                    K[i, j] = spectrum_kernel_on_sorted_keys(keys1[i], keys2[j])
    return K
