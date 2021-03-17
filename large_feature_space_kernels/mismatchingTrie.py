
import numpy as np

class MismatchingTrie(object):

    def __init__(self, m, n):
        self.root = MismatchingTrieNode("", m=m, n=n)
        self.K = None

    def add_word(self, word, index_sample=None):
        self.root.add_word(word, mismatch=0, index_sample=index_sample)

    def add_sentence(self, sentence, k, index_sample=None):
        for i in range(len(sentence)-k):
            self.add_word(sentence[i:i+k], index_sample=index_sample)

    def add_data(self, data, k):
        assert len(data) == self.root.n
        for index_sample, sentence in enumerate(data):
            self.add_sentence(sentence, k, index_sample=index_sample)
        return self

    def compute_kernel_matrix(self, normalize=False):
        self.K = np.zeros((self.root.n, self.root.n))
        self.root.compute_kernel_matrix(self.K)
        if normalize:
            diag = np.expand_dims(np.sqrt(np.diag(self.K)), 0)
            self.K /=  diag.T @ diag

    def build_lookup_table(self, coef):
        lookup_table = {}
        self.root.compute_lookup_table(lookup_table, coef, prefix="")
        return lookup_table


class MismatchingTrieNode(object):
    def __init__(self, val, m, n):

        self.label = val
        self._children = []

        self.count_sample = np.array([0] * n)
        self.n = n
        self.m = m
        self.vocab = ["T", "C", "G", "A"]

    @property
    def children(self):
        return [x for x in self._children]

    def _add(self, index_sample):
        self.count_sample[index_sample] += 1

    def add_child(self, value, index_sample=None):
        new = MismatchingTrieNode(value, m=self.m, n=self.n)
        new._add(index_sample=index_sample)
        self._children.append(new)
        return new

    def add_word(self, word, mismatch, index_sample=None):
        if len(word) > 1:
            s = word[0]
            word = word[1:]
            vocab_seen = []
            for c in self.children:
                vocab_seen.append(c.label)
                if c.has_label(s):
                    c._add(index_sample=index_sample)
                    c.add_word(word, mismatch, index_sample=index_sample)
                elif mismatch < self.m:
                    c._add(index_sample=index_sample)
                    c.add_word(word, mismatch+1, index_sample=index_sample)
                else:
                    pass
            if mismatch < self.m:
                for s_ in self.vocab:
                    if not(s_ in vocab_seen):
                        new_node = self.add_child(s_, index_sample=index_sample)
                        if new_node.has_label(s):
                            new_node.add_word(word, mismatch, index_sample=index_sample)
                        else:
                            new_node.add_word(word, mismatch+1, index_sample=index_sample)
                    else:
                        pass
            elif not(s in vocab_seen):
                new_node = self.add_child(s, index_sample=index_sample)
                new_node.add_word(word, mismatch, index_sample=index_sample)
            else:
                pass
        else:
            s = word
            vocab_seen = []
            for c in self.children:
                vocab_seen.append(c.label)
                if c.has_label(s):
                    c._add(index_sample=index_sample)
                elif mismatch < self.m:
                    c._add(index_sample=index_sample)
                else:
                    pass
            if mismatch < self.m:
                for s_ in self.vocab:
                    if not(s_ in vocab_seen):
                        self.add_child(s_, index_sample=index_sample)
            elif not(s in vocab_seen):
                self.add_child(s, index_sample=index_sample)
            else:
                pass

    def has_label(self, letter):
        return self.label == letter

    def compute_kernel_matrix(self, K):
        if self.children == []:
            K += np.array([self.count_sample]).T @ np.array([self.count_sample])
        else:
            for c in self.children:
                c.compute_kernel_matrix(K)

    def compute_lookup_table(self, lookup_table, coef, prefix):
        if self.children == []:
            lookup_table[prefix + self.label] = np.sum(coef * self.count_sample)
        else:
            for c in self.children:
                c.compute_lookup_table(lookup_table, coef, prefix + self.label)
