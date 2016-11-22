import re
import os
import sys
import random

import numpy as np
import numpy.random as npr
import scipy.stats as spsp

# Generates a set of documents using the LDA generative model
# Up to four topics allowed, with words in wordlist[1-4].txt.
# Wordlist documents taken from http://www.enchantedlearning.com/wordlist/

longestDocumentWordCount = 1000

def gen_wordcount(maxcount):
    x = npr.poisson(lam=8.0)
    return max(x,maxcount)

def gen_alpha_param(dim):
    """Generate a dim-dimensional vector to serve as alpha parameter."""
    return npr.random.rand(dim)

def get_true_index(inlist):
    """Get indices where inlist is true, throwing exception if too many true values"""
    retval = [ index for index,predicate in enumerate(inlist) if predicate ]

    if len(retval) > 1:
        raise ValueError("Too many indices were true!")

    return retval[0]

def get_words():
    """Gets the four word sets by reading from file."""

    filenames = ["wordlist1.txt", "wordlist2.txt", "wordlist3.txt", "wordlist4.txt"]

    wordbags = []

    for filename in filenames:
        with open(filename,'r') as infile:
            bagOfWords = [ word.rstrip() for word in infile.readlines() ]
            wordbags.append(bagOfWords)

    return wordbags

def gen_document_lda(documentName, alpha, bow):
    """Generate a single document using the LDA model."""

    # Generate document parameters
    numWords = gen_wordcount(longestDocumentWordCount)
    theta = npr.dirichlet(alpha)

    with open(documentName, 'w') as outfile:
        for word in range(numWords):
            topicVector = npr.multinomial(1, theta)
            topicIndex = get_true_index(topicVector == 1)

            # Assume that for a topic with n words, the multinomial parameters
            # are uniform, i.e. a vector of n elements, with each element = 1/n
            wordlist = bow[topicIndex]
            wordlistSize = len(wordlist)
            multinomParameter = np.array([ 1/wordlistSize for n in range(wordlistSize) ])

            wordVector = npr.multinomial(1,multinomParameter)
            wordIndex = get_true_index(wordVector == 1)

            assert type(wordIndex) == int

            outword = wordlist[wordIndex]
            outfile.write(outword)
            outfile.write(" ")


def main(args):
    if len(args) < 3:
        print("Usage: %s <output dir> <num articles>" % args[0])
        sys.exit(1)

    outputdir = args[1]
    ndoc = int(args[2])

    words = get_words()
    alpha = np.array([0.25,0.25,0.25,0.25])

    for i in range(ndoc):
        print("Generating document " + str(i))
        fname = "lda_out_" + str(i) + ".txt"
        outname = os.path.join(outputdir,fname)
        gen_document_lda(outname, alpha, words)

if __name__ == "__main__":
    main(sys.argv)
