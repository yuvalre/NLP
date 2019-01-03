from nltk.corpus import dependency_treebank
import numpy as np
from numpy import random
from collections import namedtuple
from Chu_Liu_Edmonds_algorithm import Arc, min_spanning_arborescence

BasicArc = namedtuple('BasicArc', ('tail', 'head'))


class SparseVector(object):
    """
    Class representing a sparse vector with the following operators:
        - addition of two SparseVectors
        - subtraction of one SparseVector from another
        - multiplication a SparseVector by a scalar
        - summation of the values of a SparseVector over given indices
    """
    def __init__(self, size, data=None):
        self.size = size
        if data is None:
            self.dict = dict()
        else:
            self.dict = data

    def sum_over_indices(self, ind):
        result = 0
        for i in ind:
            result += self.dict.get(i, 0)
        return result

    def add(self, other):
        result = self.dict.copy()
        result.update(other.dict)
        for i in set(self.dict.keys()) & set(other.dict.keys()):
            result[i] += self.dict[i]
        return SparseVector(self.size, result)

    def multiply(self, scalar):
        result = dict()
        for k, v in self.dict.items():
            result[k] = v * scalar
        return SparseVector(self.size, result)

    def subtract(self, other):
        return self.add(other.multiply(-1))

    def __getitem__(self, key):
        return self.dict.get(key, 0)

    def __setitem__(self, key, val):
        self.dict[key] = val


def find_mst(sentence, phi, theta):
    """
    Finds a Maximal Spanning Tree for the sentence, weighted by theta and phi.
    """
    n = len(sentence.nodes)
    arcs = []
    # Create a complete graph (minus the edges going into the root):
    for i in range(n):
        for j in range(1, n):
            weight = -1 * theta.sum_over_indices(phi(i, j, sentence, indices=True))
            arcs.append(Arc(j, weight, i))
    # Find MST of graph:
    return min_spanning_arborescence(arcs, 0)


def sum_phi_over_tree(tree, phi):
    """
    Sums phi(e) over each edge e in the tree.
    """
    result = phi(0, tree.root['address'], tree)
    queue = [tree.root]
    while queue:
        node = queue.pop()
        for d in node['deps'].values():
            for dep in d:
                dep = tree.get_by_address(dep)
                result = result.add(phi(node['address'], dep['address'], tree))
                queue.append(dep)
    return result


def sum_phi_over_mst(sentence, phi, theta):
    """
    Sums phi(e) over each edge e in the MST created from the sentence, weighted by phi & theta.
    """
    n = len(sentence.nodes)
    mst = find_mst(sentence, phi, theta)
    result = phi(mst[1].head, mst[1].tail, sentence)
    for i in range(2, n):
        result = result.add(phi(mst[i].head, mst[i].tail, sentence))
    return result


def get_phi1(word2i, pos2i):
    """
    Creates a feature function phi with the following features:
    - word bigrams
    - PoS bigrams
    :param word2i: a dictionary that maps each word in the vocabulary to an index in range(len(words))
    :param pos2i: a dictionary that maps each PoS to an index in range(len(pos_tags))
    :return: feature function phi
    """
    n_words = len(word2i)
    n_pos = len(pos2i)
    n_features = n_words ** 2 + n_pos ** 2
    root_node = {'word': 'ROOT', 'tag': 'ROOT'}
    word_pair2i = np.arange(n_words ** 2).reshape(n_words, n_words)
    pos_pair2i = np.arange(n_words ** 2, n_words ** 2 + n_pos ** 2).reshape(n_pos, n_pos)

    def phi(u, v, s, indices=False):
        """
        :param u: index of word/node u in sentence s.
        :param v: index of word/node v in sentence s.
        :param s: sentence
        :param indices: if True, returns a list with the indices where phi(u, v, s) == 1.
        :return: feature vector phi(u, v, s)
        """
        u = s.get_by_address(u)
        v = s.get_by_address(v)
        if indices:
            return [word_pair2i[word2i[u['word']], word2i[v['word']]],
                    pos_pair2i[pos2i[u['tag']], pos2i[v['tag']]]]
        else:
            result = SparseVector(n_features)
            result[word_pair2i[word2i[u['word']], word2i[v['word']]]] = 1
            result[pos_pair2i[pos2i[u['tag']], pos2i[v['tag']]]] = 1
            return result

    return phi, n_features


def get_phi2(word2i, pos2i):
    """
    Creates a feature function phi with the following features:
    - word bigrams
    - PoS bigrams
    - distance
    :param word2i: a dictionary that maps each word in the vocabulary to an index in range(len(words))
    :param pos2i: a dictionary that maps each PoS to an index in range(len(pos_tags))
    :return: feature function phi
    """
    n_words = len(word2i)
    n_pos = len(pos2i)
    n_features = n_words ** 2 + n_pos ** 2
    root_node = {'word': 'ROOT', 'tag': 'ROOT'}
    word_pair2i = np.arange(n_words ** 2).reshape(n_words, n_words)
    pos_pair2i = np.arange(n_words ** 2, n_words ** 2 + n_pos ** 2).reshape(n_pos, n_pos)
    dist2i = {0: n_features, 1: n_features + 1, 2: n_features + 2, 3: n_features + 3}
    n_features += 4

    def phi(u, v, s, indices=False):
        """
        :param u: index of word/node u in sentence s.
        :param v: index of word/node v in sentence s.
        :param s: sentence
        :param indices: if True, returns a list with the indices where phi(u, v, s) == 1.
        :return: feature vector phi(u, v, s)
        """
        dist_i = None
        if v <= u:
            pass
        elif v == u + 1:
            dist_i = dist2i[0]
        elif v == u + 2:
            dist_i = dist2i[1]
        elif v == u + 3:
            dist_i = dist2i[2]
        elif v > u:
            dist_i = dist2i[3]
        u = s.get_by_address(u)
        v = s.get_by_address(v)
        if indices:
            if dist_i is None:
                return [word_pair2i[word2i[u['word']], word2i[v['word']]],
                        pos_pair2i[pos2i[u['tag']], pos2i[v['tag']]]]
            else:
                return [word_pair2i[word2i[u['word']], word2i[v['word']]],
                        pos_pair2i[pos2i[u['tag']], pos2i[v['tag']]],
                        dist_i]
        else:
            result = SparseVector(n_features)
            result[word_pair2i[word2i[u['word']], word2i[v['word']]]] = 1
            result[pos_pair2i[pos2i[u['tag']], pos2i[v['tag']]]] = 1
            result[dist_i] = 1

            return result

    return phi, n_features


def perceptron(training_set, phi, n_features, epochs=2, eta=1):
    """
    Learns theta with the structured averaged perceptron algorithm.
    """
    n = len(training_set)
    theta = SparseVector(n_features)
    sum_theta = theta
    total = n * epochs
    percent = total / 200
    for k in range(epochs):
        for i in range(n):
            if i % 500 == 0:
                print(str(round((k * n + i) * 100 / (epochs * n), 2)) + '% complete')
            tree = training_set[i]
            s = tree.nodes
            update = sum_phi_over_tree(tree, phi).subtract(sum_phi_over_mst(tree, phi, theta))
            if eta != 1:
                updata = update.multiply(eta)
            theta = theta.add(update)
            sum_theta = sum_theta.add(theta)
    return sum_theta.multiply(1 / (epochs * n))


def eval_sentence(tree, phi, theta):
    """
    computes the unlabeled attachment score of theta on a single sentence.
    """
    # find arcs of MST by phi & theta
    n = len(tree.nodes)
    mst_arcs = find_mst(tree, phi, theta)

    # find arcs of gold standard tree
    queue = [tree.root]
    gold_arcs = [BasicArc(tree.root['address'], 0)]
    while queue:
        node = queue.pop()
        for d in node['deps'].values():
            for dep in d:
                dep = tree.get_by_address(dep)
                gold_arcs.append(BasicArc(dep['address'], node['address']))
                queue.append(dep)

    # compute score
    diff_n = 0
    for arc in gold_arcs:
        if mst_arcs[arc.tail].head != arc.head:
            diff_n += 1
        else:
            pass
    return 1 - (diff_n / (n - 1))


def eval_theta(test_set, phi, theta):
    """
    computes the unlabeled attachment score of theta on the test set.
    """
    n = len(test_set)
    score = 0
    for i in range(n):
        score += eval_sentence(test_set[i], phi, theta)
    return score / n


if __name__ == '__main__':
    data = dependency_treebank.parsed_sents()

    # build vocabulary and pos-tags list
    for tree in data:
        tree.get_by_address(0)['word'] = 'ROOT'
        tree.get_by_address(0)['tag'] = 'ROOT'
    words = {tree.get_by_address(i)['word'] for tree in data for i in range(len(tree.nodes))}
    words = sorted(list(words))
    pos_tags = {tree.get_by_address(i)['tag'] for tree in data for i in range(len(tree.nodes))}
    pos_tags = sorted(list(pos_tags))

    word2i = {word: i for (i, word) in enumerate(words)}
    pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}

    # shuffle data before building training & test sets
    SHUFFLE = False
    if SHUFFLE:
        data = list(data)
        random.shuffle(data)

    # build training & data sets, vocabulary and PoS tags list
    training_set_last_i = int(len(data) * 0.9)
    training_set = data[:training_set_last_i]
    test_set = data[training_set_last_i:-1]

    # train & test first feature function
    print('Without distance features:')
    phi, n_features = get_phi1(word2i, pos2i)
    print('Learning theta...')
    theta = perceptron(training_set, phi, n_features)
    print('Finished learning theta. Evaluating...')
    print('Score: ' + str(eval_theta(test_set, phi, theta)) + '\n')

    # train & test second feature function
    print('With distance features:')
    phi, n_features = get_phi2(word2i, pos2i)
    print('Learning theta...')
    theta = perceptron(training_set, phi, n_features)
    print('Finished learning theta. Evaluating...')
    print('Score: ' + str(eval_theta(test_set, phi, theta)))
