# encoding utf-8
#
# author: James Bain
# maintainer: James Bain <jamescbain@gmail.com>

import numpy as np


class EncodingProletariat(object):
    """A Encoding Class for Text Generation via LSTM

    This encoding class provides the components for pre-processing text data for training a neural net designed to
    generate texts. Text data must be in the format of a python list where each item in the list is either a word or
    punctuation.

    Parameters
    ----------
    corpus_list: list
        List of words in the corpus.

    num_inputs: int
        The number of input values per row.

    lowercase: bool
        Option to lowercase all words int he corpus_list.

    stopwords_list: list
        List of stopwords to remove from the corpus_list.
    """
    def __init__(self, corpus_list, num_inputs, lowercase=False, stopwords_list=None):
        self.corpus_list = corpus_list
        self.num_inputs = num_inputs
        self.preprocessed = self._preprocess(self.corpus_list, lowercase=lowercase, stopwords_list=stopwords_list)
        self.vocab_dict, self.reverse_dict = self._create_dictionary(self.preprocessed)
        self.encoded_list = self._encode_list(self.preprocessed, self.vocab_dict)
        self.encodings_x, self.encodings_y = self._configure_arrays(self.encoded_list, self.num_inputs)

    def _preprocess(self, corpus_list, lowercase=False, stopwords_list=None):
        """Preprocess the Corpus List.

        Provides some simple preprocessing steps that could be beneficial for training purposes. This includes an option
        to lowercase all words in the corpus list and to remove stopwords.

        Parameters
        ----------
        corpus_list: list
            List of words in the corpus.

        lowercase: bool
            Option to lowercase all words int he corpus_list.

        stopwords_list: list
            List of stopwords to remove from the corpus_list.

        Returns
        -------
        corpus_list: list
            The preprocessed corpus_list.
        """

        # lowercases words in corpus
        if lowercase:
            corpus_list = [w.lower() for w in corpus_list]

        # removes stopwords
        if stopwords_list is not None:
            corpus_list = [w for w in corpus_list if w not in stopwords_list]

        return corpus_list

    def _create_dictionary(self, preprocessed_list):
        """Create a Vocabulary Dictionary.

        Create a dictionary of the vocab from a list of words in a corpus. This function
        all so provides the option to preprocess on the fly.

        Parameters
        ----------
        preprocessed_list: list
            List of preprocessed words in the corpus.

        Returns
        -------
        vocab_dict: tuple
            A vocabulary dictionary => {word: int} and a reverse dictionary => {int: word}.
        """

        uniq_words = list(set(preprocessed_list))
        word_indexes = list(range(0, len(uniq_words)))

        vocab_dict = dict(zip(uniq_words, word_indexes))
        reverse_dict = dict(zip(word_indexes, uniq_words))

        return vocab_dict, reverse_dict

    def _encode_list(self, preprocessed_list, vocab_dict):
        """Encode Vocabulary List.

        Encodes the preprocessed text using the vocabulary dict.

        Parameters
        ----------
        preprocessed_list: list
            The preprocessed words int he corpus.

        vocab_dict: dict
            The vocabulary dict => {word: int}.

        Returns
        -------
        encoded_list: list
            The encoded version of the text list.
        """
        encoded_list = [vocab_dict[w] for w in preprocessed_list]

        return encoded_list

    def _configure_arrays(self, encoded_list, num_inputs):
        """Configure the Encoded Data into a 2-dimensional Array

        Creates a 2-dimensional array from the encoded data of an arbitrary number of inputs. Each row contains
        `num_inputs` + 1 values where the last value in each row represents the target value and the ones previous the
        inputs. These are just shifting rows where the first `num_inputs` are the first inputs in row 1 and then for row
        2 the row shifts over by 1.

        Parameters
        ----------
        encoded_list: list
            A list of encoded words.

        num_inputs: int
            The number of input values per row.

        Returns
        -------
        tuple
            A tuple of a 2-dimensional numpy.array representative of the inputs and a 1-dimensional
            numpy.array representing the targets.
        """
        config_lists_x = []
        config_lists_y = []
        for i in range(0, len(encoded_list) - num_inputs):
            config_lists_x.append(list(encoded_list[i: i + num_inputs]))
            config_lists_y.append(encoded_list[i + num_inputs])

        return np.array(config_lists_x), np.array(config_lists_y)