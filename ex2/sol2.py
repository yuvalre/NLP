from nltk.corpus import brown
import random
from string import digits
import numpy as np
import pandas as pd

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
OTHER_WORD = '*OTHER_WORD*'

PSUEDOWORDS = {TWO_DIGIT_NUM, FOUR_DIGIT_NUM, DIGIT_AND_ALPHA, DIGIT_AND_DASH, DIGIT_AND_SLASH, DIGIT_AND_COMMA,
               DIGIT_AND_PERIOD, OTHER_NUMBER, ALL_CAPS, CAP_PERIOD, FIRST_WORD, INIT_CAP, OTHER_WORD}

PREFIX_ANTI = '*ANTI*'
PREFIX_DIS = '*DIS*'
PREFIX_UNDER = '*UNDER*'
PREFIX_UN = '*UN*'
PREFIX_SEMI = '*SEMI*'
PREFIX_NON = '*NON*'
PREFIX_MID = '*MID*'
PREFIX_PRE = '*PRE*'

PREFIXES = [PREFIX_ANTI, PREFIX_DIS, PREFIX_UNDER,
            PREFIX_UN, PREFIX_SEMI, PREFIX_NON, PREFIX_MID, PREFIX_PRE]
PREFIXES_STRINGS = ['anti', 'dis', 'under', 'un', 'semi', 'now', 'mid', 'pre']

SUFFIX_ABLE = '*SUFFIX_ABLE*'
SUFFIX_IBLE = '*SUFFIX_IBLE*'
SUFFIX_AL = '*SUFFIX_AL*'
SUFFIX_IAL = '*SUFFIX_IAL*'
SUFFIX_EN = '*SUFFIX_EN*'
SUFFIX_ER = '*SUFFIX_ER*'
SUFFIX_EST = '*SUFFIX_EST*'
SUFFIX_FUL = '*SUFFIX_FUL*'
SUFFIX_IC = '*SUFFIX_IC*'
SUFFIX_ITIVE = '*SUFFIX_ITIVE*'
SUFFIX_ATIVE = '*SUFFIX_ATIVE*'
SUFFIX_IVE = '*SUFFIX_IVE*'
SUFFIX_LESS = '*SUFFIX_LESS*'
SUFFIX_LY = '*SUFFIX_LY*'
SUFFIX_MENT = '*SUFFIX_MENT*'
SUFFIX_NESS = '*SUFFIX_NESS*'


SUFFIXES = [SUFFIX_ABLE, SUFFIX_IBLE, SUFFIX_IAL, SUFFIX_EN, SUFFIX_ER, SUFFIX_AL,
            SUFFIX_EST, SUFFIX_FUL, SUFFIX_IC,
            SUFFIX_ITIVE, SUFFIX_ATIVE, SUFFIX_IVE, SUFFIX_LESS, SUFFIX_LY,
            SUFFIX_MENT, SUFFIX_NESS]
SUFFIXES_STRINGS = ['able', 'ible', 'ial', 'en', 'er', 'al', 'est', 'ful', 'ic',
                    'itive', 'ative', 'ive', 'less', 'ly', 'ment', 'ness']

PSUEDOWORDS |= set(SUFFIXES)
PSUEDOWORDS |= set(PREFIXES)


def get_psuedoword(word, t):
    '''
    return the psuedoword matching the given word.
    :param word: string representing a word in a sentence
    :param t: the word's location in the sentence
    :return: psuedoword matching the given word
    '''
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
    if word.isnumeric():
        return OTHER_NUMBER
    if all(c in DIGITS_SET.union('-') for c in word):
        return DIGIT_AND_DASH
    if all(c in DIGITS_SET.union('/') for c in word):
        return DIGIT_AND_SLASH
    if all(c in DIGITS_SET.union(',') for c in word):
        return DIGIT_AND_COMMA
    if all(c in DIGITS_SET.union('.') for c in word):
        return DIGIT_AND_PERIOD

    for i in range(len(PREFIXES)):
        if word.endswith(PREFIXES_STRINGS[i]):
            return PREFIXES[i]

    for i in range(len(SUFFIXES)):
        if word.endswith(SUFFIXES_STRINGS[i]):
            return SUFFIXES[i]

    return OTHER_WORD


def replace_with_psuedowords(data, n):
    '''
    computes the frequency of each word in the data set and replaces all low frequency
    words (occuring <= n times in the data set) with psuedowords.
    :param data: data set
    :param n: frequency threshold
    :return: data set after processing
    '''
    word_count = dict()
    for sentence in data:
        for word, pos_tag in sentence:
            word_count[word] = word_count.get(word, 0) + 1
    processed_data = []
    words = set()
    rare_words = set()
    for i, sentence in enumerate(data):
        for t in range(len(sentence)):
            if word_count[sentence[t][0]] <= n:
                rare_words.add(sentence[t][0])
                sentence[t] = (get_psuedoword(sentence[t][0], t), sentence[t][1])
            else:
                words.add(sentence[t][0])
            processed_data.append(sentence)

    words = list(words | PSUEDOWORDS)
    return processed_data, words, rare_words


class Baseline(object):
    '''
    Most likely tag baseline
    '''

    def __init__(self, training_data, words, pos_tags):
        '''
        :param training_data: training data.
        :param words: list of all words in the training data.
        :param pos_tags: list of all PoS tags in the data (including the PoS tags in
               the test set, for calculating the confusion matrix).
        '''
        self.words = words
        self.pos_tags = pos_tags
        self.pos_size = len(self.pos_tags)
        self.words_size = len(self.words)
        self.word2i = {word: i for (i, word) in enumerate(self.words)}
        self.pos2i = {pos: i for (i, pos) in enumerate(self.pos_tags)}
        self.word2pos = self.train(training_data)

    def train(self, data):
        '''
        train the model.
        :param data: training data
        :return: a dictionary mapping from word to its most likely tag
        '''
        # calculate P(tag | word) for every tag and every word
        tag_freq = np.zeros(self.pos_size)
        emissions = np.zeros((self.pos_size, self.words_size))
        for sentence in data:
            for word, pos_tag in sentence:
                tag_freq[self.pos2i[pos_tag]] += 1
                emissions[self.pos2i[pos_tag], self.word2i[word]] += 1
        emissions /= tag_freq.reshape(-1, 1)
        tag_freq /= np.sum(tag_freq)
        np.nan_to_num(emissions, copy=False)

        # create word2pos dict - word to its most likely PoS tag
        word2pos = np.argmax(tag_freq.reshape(-1, 1) * emissions, axis=0)
        word2pos = np.array(pos_tags)[word2pos]
        word2pos = {self.words[i]: word2pos[i] for i in range(self.words_size)}

        return word2pos

    def test_error(self, test_set, print_results=True):
        '''
        compute test error.
        :param test_set: test set
        :param print_results: To print or not to print
        :return: a tuple: total error, known words error, unknown words error
        '''
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
                  '\nUnknown words error:', unknown_words_error,
                  '\n')
        return total_error, known_words_error, unknown_words_error


class BigramHMM(object):
    '''
    Bigram HMM models (with options to use add-one smoothing and psuedo-words).
    '''

    def __init__(self, training_data, words, pos_tags,
                 tag_unknown_words='uniform',
                 add_one=False,
                 psuedowords=False, n=2):
        '''
        :param training_data: training data.
        :param words: list of all words in the training data.
        :param pos_tags: list of all PoS tags in the data (including the PoS tags in
               the test set, for calculating the confusion matrix).
        :param tag_unknown_words: How to handle unknown words in the simple HMM model,
                                  when not using add-one or psuedo-words.
                                    uniform - uniform distribution over all PoS tags
                                    NN      - tag as NN
        :param add_one: To do or not to do (add-one smoothing)
        :param psuedowords: To do or not to do (psuedo-words)
        :param n: frequency threshold for psuedo-words. only relevant if psuedowords=True
        '''
        self.tag_unknown_words = tag_unknown_words
        self.add_one = add_one
        self.psuedowords = psuedowords
        if psuedowords:
            self.n = n
            training_data, self.words, rare_words = replace_with_psuedowords(training_data, n)
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

        self.transitions, self.emissions = self.train(training_data)

    def train(self, data):
        '''
        train the model.
        :param data: training data
        :return: numpy arrays of transitions & emissions probabilities
        '''
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
        if self.add_one:
            emissions = emissions + 1
            pos_count = pos_count + self.words_size
        transitions /= states_count.reshape(-1, 1)

        emissions /= pos_count.reshape(-1, 1)
        np.nan_to_num(emissions, copy=False)  # replaces missing values with 0
        np.nan_to_num(transitions, copy=False)

        return transitions, emissions

    def get_emission_probs(self, word, unknown_words_indices, t):
        '''
        get the emission probabilities of one word.
        :param word: word from sentence
        :param unknown_words_indices: a list of unknown word indices, used for updating it
               in case this word is unknown as well (for calculating unknown words error)
        :param t: the word's location in the sentence
        :return: numpy vector of this word's emission probabilities
        '''
        if self.psuedowords:
            if word not in self.word2i:
                unknown_words_indices.add(t)
                word = get_psuedoword(word, t)
            return self.emissions[:, self.word2i[word]]

        if self.add_one:
            if word not in self.word2i:
                unknown_words_indices.add(t)
            return self.emissions[:, self.word2i.get(word, self.word2i[UNKNOWN_WORD])]

        if word not in self.word2i:
            unknown_words_indices.add(t)
            if self.tag_unknown_words == 'uniform':
                emission_probs = np.ones(self.pos_size)/self.pos_size
            elif self.tag_unknown_words == 'NN':
                selected_pos_tag_i = self.pos2i['NN']  # random.choice(self.pos_axis)
                emission_probs = np.zeros(self.pos_size)
                emission_probs[selected_pos_tag_i] = 1
            return emission_probs

        return self.emissions[:, self.word2i[word]]

    def viterbi(self, sentence):
        '''
        Viterbi algorithm for HMM
        :param sentence: sentence to tag.
        :return: A tuple:
                    sentence_tags - list of the sentence's tags
                    unknown_words_ind - indices of all of the unknown words in the sentence
        '''
        unknown_words_ind = set()
        n = len(sentence)
        paths = list(range(n - 1))
        sentence_tags = list(range(n))

        # forward pass:
        pi = self.transitions[self.pos2i[START_STATE], :] *\
             self.get_emission_probs(sentence[0], unknown_words_ind, 0)
        for i in range(1, n):
            # pi(t, v) matrix - before taking maximum over previous state w:
            pi = self.transitions * pi.reshape(-1, 1) *\
                 self.get_emission_probs(sentence[i], unknown_words_ind, i).reshape(1, -1)
            bp = np.argmax(pi, axis=0)
            # pi_t(i) column - after maximization, with emission probabilities:
            pi = pi[bp, self.pos_axis]
            paths[i-1] = bp
        pi = self.transitions[:, self.pos2i[END_STATE]] * pi

        # back tracing:
        sentence_tags[-1] = np.argmax(pi)
        for i in range(n-2, -1, -1):
            bp = paths.pop()
            sentence_tags[i] = bp[sentence_tags[i+1]]
            sentence_tags[i+1] = self.pos_tags[sentence_tags[i+1]]
        sentence_tags[0] = self.pos_tags[sentence_tags[0]]

        return sentence_tags, unknown_words_ind

    def test_error(self, test_set, print_results=True):
        '''
            compute test error.
            :param test_set: test set
            :param print_results: To print or not to print
            :return: a tuple: total error, known words error, unknown words error
        '''
        n_known_words = 0
        n_unknown_words = 0
        n_known_words_incorrect_tag = 0
        n_unknown_words_incorrect_tag = 0
        X = [[word_pos_tuple[0] for word_pos_tuple in sentence] for sentence in test_set]

        for i, sentence in enumerate(test_set):
            y_i_hat, unknown_words_ind = self.viterbi(X[i])
            for t, (word, pos_tag) in enumerate(sentence):
                if t in unknown_words_ind:
                    n_unknown_words += 1
                    if pos_tag != y_i_hat[t]:
                        n_unknown_words_incorrect_tag += 1
                else:
                    n_known_words += 1
                    if y_i_hat[t] != pos_tag:
                        n_known_words_incorrect_tag += 1

        total_error = (n_known_words_incorrect_tag + n_unknown_words_incorrect_tag) /\
                      (n_known_words + n_unknown_words)
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
        return total_error, known_words_error, unknown_words_error

    def get_confusion_mat(self, test_set):
        '''
        compute confusion matrix.
        '''
        confusion_mat = np.zeros((self.pos_size, self.pos_size))
        X = [[word_pos_tuple[0] for word_pos_tuple in sentence] for sentence in test_set]
        for i, sentence in enumerate(test_set):
            y_i_hat, unknown_words_ind = self.viterbi(X[i])
            for t, (word, pos_tag) in enumerate(sentence):
                confusion_mat[self.pos2i[pos_tag], self.pos2i[y_i_hat[t]]] += 1
        return confusion_mat


if __name__ == '__main__':
    data = brown.tagged_sents(categories=['news'])[:]
    # shuffle data before building training & test sets
    SHUFFLE = False
    if SHUFFLE:
        data = list(data)
        random.shuffle(data)

    # build training & data sets, vocabulary and PoS tags list
    training_set_last_i = int(len(data) * 0.9)
    training_set = data[:training_set_last_i]
    test_set = data[training_set_last_i:-1]
    training_words = list({word_pos_tuple[0] for sentence in training_set for word_pos_tuple in sentence})
    pos_tags = sorted(list({word_pos_tuple[1] for sentence in data for word_pos_tuple in sentence}))

    # train and test all models
    models = list()
    models_names = list()

    models_names.append('Baseline')
    models.append(Baseline(training_set, training_words, pos_tags))
    models_names.append('Bigram HMM -  Unknown words as NN')
    models.append(BigramHMM(training_set, training_words, pos_tags, tag_unknown_words='NN'))
    models_names.append('Bigram HMM -  Unknown words with uniform emission')
    models.append(BigramHMM(training_set, training_words, pos_tags))
    models_names.append('Bigram HMM + add-1')
    models.append(BigramHMM(training_set, training_words, pos_tags, add_one=True))
    models_names.append('Bigram HMM + psuedo, freq=2')
    models.append(BigramHMM(training_set, training_words, pos_tags, psuedowords=True, n=2))
    models_names.append('Bigram HMM + psuedo + add-1, freq=2')
    models.append(BigramHMM(training_set, training_words, pos_tags,
                            psuedowords=True, n=2, add_one=True))
    models_names.append('Bigram HMM + psuedo, freq=1')
    models.append(BigramHMM(training_set, training_words, pos_tags, psuedowords=True, n=1))
    models_names.append('Bigram HMM + psuedo + add-1, freq=1')
    models.append(BigramHMM(training_set, training_words, pos_tags,
                            psuedowords=True, n=1, add_one=True))

    # compare frequency parameter
    pwords_models = list()
    pwords_models_names = list()
    max_freq = 5
    for n in range(1, max_freq + 1):
        pwords_models_names.append('Bigram HMM + psuedo, freq=' + str(n))
        pwords_models.append(BigramHMM(training_set, training_words, pos_tags, psuedowords=True, n=n))
    for n in range(1, max_freq + 1):
        pwords_models_names.append('Bigram HMM + psuedo + add-1, freq=' + str(n))
        pwords_models.append(BigramHMM(training_set, training_words, pos_tags,
                                       psuedowords=True, n=n, add_one=True))

    # dump error charts
    index = ['Total error', 'Known words error', 'Unknown words error']
    errors = [model.test_error(test_set, print_results=True) for model in models]
    models_errors = pd.DataFrame(np.array(errors).T, columns=models_names, index=index)

    errors = [model.test_error(test_set, print_results=False) for model in pwords_models]
    pwords_errors = pd.DataFrame(np.array(errors).T, columns=pwords_models_names, index=index)

    models_errors.to_csv('models_errors.csv')
    pwords_errors.to_csv('pwords_frequency.csv')

    # dump confusiom matrix to file for investigating the most frequent errors
    confusion_mat_model = models[-1]
    confusion_mat = confusion_mat_model.get_confusion_mat(test_set)
    pd.DataFrame(confusion_mat, columns=confusion_mat_model.pos_tags,
                 index=confusion_mat_model.pos_tags).to_csv('confusion_matrix.csv')

    # create a list of most frequent errors
    np.fill_diagonal(confusion_mat, 0)
    confusion_err_feq = confusion_mat / confusion_mat.sum()
    frequent_errors_i = np.argwhere(confusion_err_feq > 0.005).tolist()

    frequent_errors = list()
    for true_pos_i, predicted_pos_i in frequent_errors_i:
        frequent_errors.append((confusion_mat_model.pos_tags[true_pos_i],
                                confusion_mat_model.pos_tags[predicted_pos_i],
                                confusion_mat[true_pos_i, predicted_pos_i]))
    pd.DataFrame(np.array(frequent_errors), columns=['True tag', 'Predicted Tag', '#']).\
        to_csv('frequent_errors.csv')
