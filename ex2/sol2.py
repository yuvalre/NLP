from nltk.corpus import brown
import random
from string import digits
import numpy as np

START_STATE = 'START'
END_STATE = 'END'
UNKNOWN_WORD = '*UNKNOWN_WORD*'

# Pseudowords:
DIGITS_SET = set(digits)
TWO_DIGIT_NUM = '*TWO_DIGIT_NUM*'
FOUR_DIGIT_NUM = '*FOUR_DIGIT_NUM*'
DIGIT_AND_ALPHA = '*DIGIT_AND_ALPHA*'
DIGIT_AND_DASH = '*DIGIT_AND_DASH*'
DIGIT_AND_SLASH = '*DIGIT_AND_SLASH*'
DIGIT_AND_COMMA = '*DIGIT_AND_COMMA*'
DIGIT_AND_PERIOD = '*DIGIT_AND_PERIOD*'
OTHER_NUMBER = '*OTHER_NUMBER*'
ALL_CAPS = '*ALL_CAPS*'
CAP_PERIOD = '*CAP_PERIOD*'
FIRST_WORD = '*FIRST_WORD*'
INIT_CAP = '*INIT_CAP*'


PSUEDOWORDS = {TWO_DIGIT_NUM, FOUR_DIGIT_NUM, DIGIT_AND_ALPHA, DIGIT_AND_DASH, DIGIT_AND_SLASH, DIGIT_AND_COMMA,
               DIGIT_AND_PERIOD, OTHER_NUMBER, ALL_CAPS, CAP_PERIOD, FIRST_WORD, INIT_CAP, UNKNOWN_WORD}


def get_psuedoword(word, t):
    if word.isupper():
        if len(word) == 2 and word[1] == '.':
            return CAP_PERIOD
        return ALL_CAPS
    if word[0].isupper():
        if t == 0:
            return FIRST_WORD
        return INIT_CAP
    if word.isdigit():
        if len(word) == 2:
            return TWO_DIGIT_NUM
        elif len(word) == 4:
            return FOUR_DIGIT_NUM
    if word.isalnum():
        return DIGIT_AND_ALPHA
    if all(c in DIGITS_SET.union('-') for c in word):
        return DIGIT_AND_DASH
    if all(c in DIGITS_SET.union('/') for c in word):
        return DIGIT_AND_SLASH
    if all(c in DIGITS_SET.union(',') for c in word):
        return DIGIT_AND_COMMA
    if all(c in DIGITS_SET.union('.') for c in word):
        return DIGIT_AND_PERIOD
    if word.isnumeric():
        return OTHER_NUMBER
    return UNKNOWN_WORD


def replace_with_psuedowords(data, n):
    word_count = dict()
    for sentence in data:
        for word, pos_tag in sentence:
            word_count[word] = word_count.get(word, 0) + 1
    processed_data = []
    words = set()
    for i, sentence in enumerate(data):
        for t in range(len(sentence)):
            if word_count[sentence[t][0]] <= n:
                sentence[t] = (get_psuedoword(sentence[t][0], t), sentence[t][1])
            else:
                words.add(sentence[t][0])
            processed_data.append(sentence)

    words = list(words | PSUEDOWORDS)
    return processed_data, words


class Baseline(object):

    def __init__(self, training_data, words, pos_tags):
        self.words = words
        self.pos_tags = pos_tags
        self.pos_size = len(self.pos_tags)
        self.words_size = len(self.words)
        self.word2i = {word: i for (i, word) in enumerate(self.words)}
        self.pos2i = {pos: i for (i, pos) in enumerate(self.pos_tags)}
        self.word2pos = self.train(training_data)

    def train(self, data):
        # calculate P(tag | word) for every tag and every word
        tag_freq = np.zeros(self.pos_size)
        emissions = np.zeros((self.pos_size, self.words_size))
        for sentence in data:
            for word, pos_tag in sentence:
                tag_freq[self.pos2i[pos_tag]] += 1
                emissions[self.pos2i[pos_tag], self.word2i[word]] += 1
        emissions /= tag_freq.reshape(-1, 1)
        tag_freq /= np.sum(tag_freq)

        # create word2pos dict - word to its most likely PoS tag
        word2pos = np.argmax(tag_freq.reshape(-1, 1) * emissions, axis=0)
        word2pos = np.array(pos_tags)[word2pos]
        word2pos = {self.words[i]: word2pos[i] for i in range(self.words_size)}

        return word2pos

    def test_error(self, test_set, print_results=True):
        n_known_words = 0
        n_unknown_words = 0
        n_known_words_incorrect_tag = 0
        n_unknown_words_incorrect_tag = 0
        for sentence in test_set:
            for word, pos_tag in sentence:
                if word not in self.word2i:
                    n_unknown_words += 1
                    if pos_tag != 'NN':
                        n_unknown_words_incorrect_tag += 1
                else:
                    n_known_words += 1
                    if self.word2pos[word] != pos_tag:
                        n_known_words_incorrect_tag += 1

        total_error = (n_known_words_incorrect_tag + n_unknown_words_incorrect_tag) / (n_known_words + n_unknown_words)
        known_words_error = n_known_words_incorrect_tag / n_known_words
        unknown_words_error = n_unknown_words_incorrect_tag / n_unknown_words

        if print_results:
            print('MLE Baseline model:')
            print('Total error:', total_error,
                  '\nKnown words error:', known_words_error,
                  '\nUnknown words error:', unknown_words_error)
        return total_error, known_words_error, unknown_words_error


class BigramHMM(object):

    def __init__(self, training_data, words, pos_tags, add_one=False, psuedowords=False, n=2,
                 compute_confusion=False):
        self.add_one = add_one
        self.psuedowords = psuedowords
        self.compute_confusion = compute_confusion
        if psuedowords:
            self.n = n
            training_data, self.words = replace_with_psuedowords(training_data, n)
        elif add_one:
            self.words = words + [UNKNOWN_WORD]
        else:
            self.words = words
        self.pos_tags = pos_tags + [START_STATE, END_STATE]
        self.pos_size = len(self.pos_tags)
        self.pos_axis = list(range(self.pos_size))
        self.words_size = len(self.words)
        self.word2i = {word: i for (i, word) in enumerate(self.words)}
        self.pos2i = {pos: i for (i, pos) in enumerate(self.pos_tags)}
        self.confusion_mat = np.zeros((self.pos_size, self.pos_size))

        self.transitions, self.emissions = self.train(training_data)
        self.log_transitions, self.log_emissions = np.log(self.transitions), np.log(self.emissions)

    def train(self, data):
        # compute transition & emission probabilities
        transitions = np.zeros((self.pos_size, self.pos_size))
        emissions = np.zeros((self.pos_size, self.words_size))
        pos_count = np.zeros(self.pos_size)

        for sentence in data:
            prev_pos_i = self.pos2i[START_STATE]
            pos_count[prev_pos_i] += 1
            for word, pos_tag in sentence:
                curr_pos_i = self.pos2i[pos_tag]
                pos_count[curr_pos_i] += 1
                emissions[curr_pos_i, self.word2i[word]] += 1
                transitions[prev_pos_i, curr_pos_i] += 1
                prev_pos_i = curr_pos_i
            transitions[prev_pos_i, self.pos2i[END_STATE]] += 1
            pos_count[self.pos2i[END_STATE]] += 1

        states_count = transitions.sum(axis=1)
        states_count[self.pos2i[END_STATE]] = 1  # to avoid division by zero (no state follows END_STATE)
        if self.add_one:
            emissions = emissions + 1
            pos_count = pos_count + self.words_size
        transitions /= states_count.reshape(-1, 1)
        emissions /= pos_count.reshape(-1, 1)

        return transitions, emissions

    def get_emission_probs(self, word, unknown_words_indices, t):
        if self.psuedowords:
            if word not in self.word2i:
                unknown_words_indices.add(t)
                word = get_psuedoword(word, t)
            return self.emissions[:, self.word2i[word]]

        if self.add_one:
            if word not in self.word2i:
                unknown_words_indices.add(t)
            return self.emissions[:, self.word2i.get(word, self.word2i[UNKNOWN_WORD])]

        # else, add_one=False, psuedo_words=False
        if word not in self.word2i:
            unknown_words_indices.add(t)
            emission_probs = np.ones(self.pos_size)/self.pos_size
            # selected_pos_tag_i = self.pos2i['NN']  # random.choice(self.pos_axis)
            # emission_probs = np.zeros(self.pos_size)
            # emission_probs[selected_pos_tag_i] = 1
            return emission_probs

        return self.emissions[:, self.word2i[word]]

    def viterbi(self, sentence):
        unknown_words_ind = set()
        n = len(sentence)
        pi = self.transitions[self.pos2i[START_STATE], :] * self.get_emission_probs(sentence[0], unknown_words_ind, 0)
        paths = list(range(n - 1))
        sentence_tags = list(range(n))

        for i in range(1, n):
            # pi(t, v) matrix - before taking maximum over previous state w:
            # (emission probabilities are not needed for maximizing pi(t, v) over w)
            pi = self.transitions * pi.reshape(-1, 1)
            bp = np.argmax(pi, axis=0)
            # pi = pi.max(axis=0) * self.get_emission_probs(sentence[i], unknown_words_ind, i)
            # pi_t(i) column - after maximization, with emission probabilities:
            pi = self.get_emission_probs(sentence[i], unknown_words_ind, i) * pi[bp, self.pos_axis]
            paths[i-1] = bp
        pi = self.transitions[:, self.pos2i[END_STATE]] * pi

        # back tracing:
        sentence_tags[-1] = np.argmax(pi)
        for i in range(n-2, 0, -1):
            bp = paths.pop()
            sentence_tags[i] = bp[sentence_tags[i+1]]
            sentence_tags[i+1] = self.pos_tags[sentence_tags[i+1]]
        sentence_tags[0] = self.pos_tags[sentence_tags[0]]
        return sentence_tags, unknown_words_ind

    def test_error(self, test_set, print_results=True):
        n_known_words = 0
        n_unknown_words = 0
        n_known_words_incorrect_tag = 0
        n_unknown_words_incorrect_tag = 0
        X = [[word_pos_tuple[0] for word_pos_tuple in sentence] for sentence in test_set]
        # y = [[word_pos_tuple[1] for word_pos_tuple in sentence] for sentence in test_set]

        for i, sentence in enumerate(test_set):
            y_i_hat, unknown_words_ind = self.viterbi(X[i])
            # print(y_i_hat)
            for t, (word, pos_tag) in enumerate(sentence):
                # if self.compute_confusion:
                    # print(self.pos2i[pos_tag])
                    # print(self.pos2i[y_i_hat[t]])
                    # self.confusion_mat[self.pos2i[pos_tag], self.pos2i[y_i_hat[t]]] += 1
                if t in unknown_words_ind:
                    n_unknown_words += 1
                    if pos_tag != y_i_hat[t]:
                        n_unknown_words_incorrect_tag += 1
                else:
                    n_known_words += 1
                    if y_i_hat[t] != pos_tag:
                        n_known_words_incorrect_tag += 1

        total_error = (n_known_words_incorrect_tag + n_unknown_words_incorrect_tag) / (n_known_words + n_unknown_words)
        known_words_error = n_known_words_incorrect_tag / n_known_words
        unknown_words_error = n_unknown_words_incorrect_tag / n_unknown_words

        if print_results:
            if self.add_one and not self.psuedowords:
                print('Bigram HMM model with add-1 smoothing:')
            elif self.psuedowords and not self.add_one:
                print('Bigram HMM model with psuedo words, n=' + str(self.n) + ':')
            elif self.psuedowords and self.add_one:
                print('Bigram HMM model with psuedo words and add-1 smoothing, n=' + str(self.n) + ':')
            else:
                print('Bigram HMM model:')
            print('Total error:', total_error,
                  '\nKnown words error:', known_words_error,
                  '\nUnknown words error:', unknown_words_error,
                  '\n')
        else:
            return total_error, known_words_error, unknown_words_error

    def get_confusion_mat(self):
        return self.confusion_mat


if __name__ == '__main__':
    SHUFFLE = False
    # if SHUFFLE:
    #     data = list(data)
    #     random.shuffle(data)
    data = brown.tagged_sents(categories=['news'])[:]
    training_set_last_i = int(len(data) * 0.9)
    training_set = data[:training_set_last_i]
    test_set = data[training_set_last_i:-1]
    training_words = list({word_pos_tuple[0] for sentence in training_set for word_pos_tuple in sentence})
    pos_tags = list({word_pos_tuple[1] for sentence in training_set for word_pos_tuple in sentence})

    models = []
    # models.append(Baseline(training_set, training_words, pos_tags))
    # models.append(BigramHMM(training_set, training_words, pos_tags))
    # models.append(BigramHMM(training_set, training_words, pos_tags, add_one=True))
    # for n in range(1, 6):
    #     models.append(BigramHMM(training_set, training_words, pos_tags, psuedowords=True, n=n))
    #     models.append(BigramHMM(training_set, training_words, pos_tags,
    #                             psuedowords=True, n=n, add_one=True, compute_confusion=True))

    test = BigramHMM(training_set, training_words, pos_tags,
                     psuedowords=True, n=1, add_one=True, compute_confusion=True)
    test.test_error(test_set)
    print(test.get_confusion_mat())







