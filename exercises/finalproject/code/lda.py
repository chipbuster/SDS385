import re
import os
import sys
import random

import numpy as np
import numpy.random as npr
import scipy.special as spsp

def init_lambda(ntopics):
    assert ntopics > 0, "The fuck you doin?"
    return npr.rand(ntopics,1)

def samp_doc(path):
    """Sample a random document from the directory."""

    fnames = os.listdir(path)
    chosenFilename = random.choice(fnames)

    return os.path.join(path,chosenFilename)

def index_words(documents):
    """Reads every document and creates an index of words.

    Documents should be a list of strings. Each string contains the contents
    of an entire document (newlines and whitespace included.)"""

    word_to_index = {}
    index_to_word = {}
    index = 0

    for docstring in documents:
        words = docstring.split()
        for word in words:
            if word in word_to_index:
                continue
            else:
                word_to_index[index] = word
                index_to_word[word] = index
                index += 1

    return (word_to_index, index_to_word)

def read_documents(path):
    """Given a path, returns document contents as generator of strings."""

    files = os.listdir(path)

    for fname in files:
        with open(os.path.join(path,fname),'r') as infile:
            yield infile.read()
