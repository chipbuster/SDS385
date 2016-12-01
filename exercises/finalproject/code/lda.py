import re
import os
import sys
import random
import math

import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import scipy.special as spsp

from copy import deepcopy

import pdb

# The algorithm in this paper is SVI, from the paper "Stochastic Variational
# Inference" by Hoffman, Blei, Wang, and Paisley, JMLR 14(2013), 1303-1347
# HoffmanF6 refers to the algorithm presented in Figure 6 of that paper.

class WordPool:
    def __init__(self, path):
        self.path = path
        (x,y) = self.index_words()
        self.wordToIndexTbl = x
        self.indexToWordTbl = y
        self.wordct = len(x)
        self.ndocs = len(os.listdir(path))

    def read_documents(self):
        """Given a path, returns document contents as generator of strings."""

        files = os.listdir(self.path)

        for fname in files:
            with open(os.path.join(self.path,fname),'r') as infile:
                yield infile.read()

    def index_words(self):
        """Reads every document and creates an index of words."""

        documents = self.read_documents()

        wordToIndex = {}
        indexToWord = {}
        index = 0
        maxlen = 0

        for docstring in documents:
            words = docstring.split()

            #if len(words) > maxlen:
            #    maxlen = len(words)
            for word in words:
                if word in wordToIndex:
                    continue
                else:
                    wordToIndex[word] = index
                    indexToWord[index] = word
                    index += 1

        return (wordToIndex, indexToWord)

    def gen_word_index_vector(self, word):
        """Generates a column vector corresponding to the word"""

        vec = np.zeros((self.wordct,1))
        index = self.wordToIndexTbl[word]
        vec[index,0] = 1

        return vec

def init_lambda(ntopics, nwords):
    """Initialize the lambda vectors (step 1 of HoffmanF6)

    There are ntopics vectors, each with nwords entries. The k-th
    lamba vector is the k-th row of the returned array."""
    assert ntopics > 0, "The fuck you doin?"
    assert nwords > 0, "The fuck you doin?"
    return np.sqrt(np.sqrt(npr.rand(ntopics,nwords)))

def normalize_l1(vector):
    """Normalize the given vector by L1 norm."""
    return vector / npla.norm(vector,1)

def samp_doc(path):
    """Sample a random document from the directory. Return a list of words."""

    fnames = os.listdir(path)
    numFiles = len(fnames)
    randIndex = random.randrange(numFiles)
    chosenFilename = fnames[randIndex]

    with open(os.path.join(path,chosenFilename), 'r') as infile:
        docContents = infile.read()

    return (randIndex,docContents.split())

def calc_logthetadk(gammavec,k):
    """Calculate E[log theta_dk] according to Figure 5 of Hoffman"""

    term1 = spsp.digamma(gammavec[k])
    term2 = np.sum(spsp.digamma(gammavec))

    return term1 - term2

def calc_logbetakv(lambdavec, k, v):
    """Calculate E[log beta_kv] according to Figure 5 of Hoffman"""

    term1 = spsp.digamma(lambdavec[k,v])
    term2 = np.sum(spsp.digamma(lambdavec[k,:]))

    return term1 - term2

def lda_documents(path, ntopics, alpha, eta, wordPool):
    """Solve LDA posterior on all documents in path."""

    nwords = wordPool.wordct
    ndocs = wordPool.ndocs

    # A set of ntopics dirichelet parameters, each nwords long
    # Each parameter is in a column of this array e.g. lParams[:,4]
    lParams = init_lambda(ntopics, nwords)
    rho = 1

    convergenceParam = 1000

    iterct = 0
    while iterct < 5e5:
        iterct += 1

        # docWords is a list of words of a single document
        docIndex,docWords = samp_doc(path)
        nDocWords = len(docWords)

        gamma = np.ones(ntopics)
        phi = np.zeros((nDocWords, ntopics))

        phi_old = deepcopy(phi)
        gamma_old = deepcopy(gamma)

        gamma_converged = False
        phi_converged = False

        while not gamma_converged or not phi_converged:
            for n, word in enumerate(docWords):
                elogtheta = np.zeros(ntopics)
                elogbeta = np.zeros(ntopics)
                wordId = wordPool.wordToIndexTbl[word]
                for k in range(ntopics):
                    elogtheta[k] = calc_logthetadk(gamma,k)
                    elogbeta[k] = calc_logbetakv(lParams,k,wordId)

                normfactor = np.mean(elogbeta + elogtheta)

                phivec = normalize_l1(np.exp(elogbeta + elogtheta - normfactor))
                phi[n,:] = phivec / np.min(phivec)
                #print(phi[n,:])
            gupdate = np.sum(phi,axis=0)
            print(gupdate)
            gamma = alpha + gupdate

            # Check gamma convergence
            deltaGamma = gamma_old - gamma
            assert np.shape(gamma_old) == np.shape(gamma)
            print(gamma_old, gamma)
            print(deltaGamma)
            dg = npla.norm(deltaGamma, 1) / np.size(deltaGamma)
            if dg < 0.01:
                gamma_converged = True
            else:
                gamma_converged = False
            gamma_old = gamma

            # Check phi convergence
            deltaPhi = phi_old - phi
            assert np.shape(phi_old) == np.shape(phi)
            dp = npla.norm(deltaPhi,1) / np.size(deltaPhi)
            if dp < 0.01:
                phi_converged = True
            else:
                phi_converged = False
            phi_old = phi

            print("deltap:", dp, dg)

        print("BREAKOUT INTO LAMBDA UPDATE")
        pdb.set_trace()
        print(phi)
        print(gamma)
        rho *= 0.7
        deltaLambda = np.zeros(np.shape(lParams)) + eta

        for k in range(ntopics):
            deltaLambda[k,:] = np.ones(np.shape(lParams[k,:])) * eta +\
                           ndocs * np.sum(phi[:,k])
        print(deltaLambda)

def lda_test_driver(path, ntopics):
    wordPool = WordPool(path)
    nwords = wordPool.wordct
    ndocs = wordPool.ndocs

    alpha = 1 / ntopics
    eta = 1 / nwords

    print("Begin calculations")
    lda_documents(path,ntopics,alpha,eta,wordPool)
