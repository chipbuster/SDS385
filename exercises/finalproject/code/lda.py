import re
import os
import sys
import random
import math
import pickle

import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import scipy.special as spsp

from copy import deepcopy

import pdb

#np.set_printoptions(threshold=np.nan)  #Uncomment to print full arrays

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
    #return npr.gamma(8,1,(ntopics,nwords))
    return npr.rand(ntopics,nwords)

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

def calc_logthetadk(gammavec,k, term2):
    """Calculate E[log theta_dk] according to Figure 5 of Hoffman"""

    term1 = spsp.digamma(gammavec[k])

    # Cached by outside computation
    # term2 = np.sum(spsp.digamma(gammavec))

    return term1 - term2

def calc_logbetakv(lambdavec, k, v, term2):
    """Calculate E[log beta_kv] according to Figure 5 of Hoffman"""

    term1 = spsp.digamma(lambdavec[k,v])

    # Cached by outside computation
    #term2 = np.sum(spsp.digamma(lambdavec[k,:]))

    return term1 - term2

def lda_documents(path, ntopics, alpha, eta, wordPool):
    """Solve LDA posterior on all documents in path."""

    nwords = wordPool.wordct
    ndocs = wordPool.ndocs

    # A set of ntopics dirichelet parameters, each nwords long
    # Each parameter is in a column of this array e.g. lParams[:,4]
    lParams = init_lambda(ntopics, nwords)
    rho = 0.9

    converged = False

    iterct = 0
    while not converged:
        iterct += 1

        # docWords is a list of words of a single document
        docIndex,docWords = samp_doc(path)
        docWordIDs = [ wordPool.wordToIndexTbl[word] for word in docWords ]
        nDocWords = len(docWords)

#        print("Document " + str(docIndex) + " has "  + str(nDocWords) + " words " +\
#              " in a vocabulary of " + str(nwords) + " words.")

        gamma = np.ones(ntopics)
        phi = np.zeros((nDocWords, ntopics))

        phi_old = deepcopy(phi)
        gamma_old = deepcopy(gamma)

        gamma_converged = False
        phi_converged = False

        elogtheta = np.zeros(ntopics)
        elogbeta = np.zeros(ntopics)


        # Conceptually, this next line goes inside the loop over enumerate(docWordIDs)
        # However, since lParams is not updated inside this loop, it is safe to cache
        # this computation and do it just once per document.
        lParamSum = np.sum(spsp.digamma(lParams),axis=1)

        while not gamma_converged or not phi_converged:
            for n, wordId in enumerate(docWordIDs):
                gammaSum = np.sum(spsp.digamma(gamma))

                assert np.size(lParamSum) == ntopics

                for k in range(ntopics):
                    elogtheta[k] = calc_logthetadk(gamma,k, gammaSum)
                    elogbeta[k] = calc_logbetakv(lParams,k,wordId,lParamSum[k])

                normfactor = np.max(elogbeta + elogtheta)

#                print(elogbeta, elogtheta)

                phivec = normalize_l1(np.exp(elogbeta + elogtheta - normfactor))
                phi[n,:] = phivec
                #print(phi[n,:])
            gupdate = np.sum(phi,axis=0)
            gamma = alpha + gupdate

            # Check gamma convergence
            deltaGamma = gamma_old - gamma
            assert np.shape(gamma_old) == np.shape(gamma)
            #print(gamma_old, gamma)
            #print(deltaGamma)
            dg = npla.norm(deltaGamma, 1) / np.size(deltaGamma)
            if dg < 1e-4:
                gamma_converged = True
            else:
                gamma_converged = False
            gamma_old = gamma

            # Check phi convergence
            deltaPhi = phi_old - phi
            assert np.shape(phi_old) == np.shape(phi)
            dp = npla.norm(deltaPhi,1) / np.size(deltaPhi)
            if dp < 1e-4:
                phi_converged = True
            else:
                phi_converged = False
            phi_old = phi

            #print("deltap:", dp, dg)

        #print("BREAKOUT INTO LAMBDA UPDATE")
        rho *= 0.95
        deltaLambda = np.zeros(np.shape(lParams)) + eta

        #print(gamma)
        #print('phi')
        #print(phi)

        for k in range(ntopics):
            for w in range(nDocWords):
                wordIndex = docWordIDs[w]
                deltaLambda[k,wordIndex] += ndocs * phi[wordIndex,k]

        assert np.shape(deltaLambda) == np.shape(lParams)

        lParams = (1 - rho) * lParams + rho * deltaLambda

        lParamDeltaMag = rho * npla.norm(deltaLambda.T) / np.size(deltaLambda)
        print(lParamDeltaMag)

        if lParamDeltaMag < 1e-4:
            print("Finished after " + str(iterct) + " global lambda updates.")
            return lParams

    print("Program did not converge after MANY iterations. This shouldn't even be possible.")
    return lParams

def genDocDistributions(lParams, ntopics, alpha, eta, docIndex, path, wordPool):
    """Generate the phi/gamma parameters for a given document.

    Once the lambda parameters have been finalized, we need to generate
    the per-document topic and word-topic distributions. Do this by
    following the same procedure as in the mainloop of LDA."""

    docName = os.listdir(path)[docIndex]
    with open(os.path.join(path,docName),'r') as infile:
        docContents = infile.read()
        docWords = docContents.split()

    docWordIDs = [ wordPool.wordToIndexTbl[word] for word in docWords ]
    nDocWords = len(docWords)

    gamma = np.ones(ntopics)
    phi = np.zeros((nDocWords, ntopics))

    phi_old = deepcopy(phi)
    gamma_old = deepcopy(gamma)

    gamma_converged = False
    phi_converged = False

    elogtheta = np.zeros(ntopics)
    elogbeta = np.zeros(ntopics)

    lParamSum = np.sum(spsp.digamma(lParams),axis=1)
    while not gamma_converged or not phi_converged:
        for n, wordId in enumerate(docWordIDs):
            gammaSum = np.sum(spsp.digamma(gamma))

            assert np.size(lParamSum) == ntopics

            for k in range(ntopics):
                elogtheta[k] = calc_logthetadk(gamma,k, gammaSum)
                elogbeta[k] = calc_logbetakv(lParams,k,wordId,lParamSum[k])

            normfactor = np.max(elogbeta + elogtheta)

            phivec = normalize_l1(np.exp(elogbeta + elogtheta - normfactor))
            phi[n,:] = phivec
            #print(phi[n,:])
        gupdate = np.sum(phi,axis=0)
        gamma = alpha + gupdate

        # Check gamma convergence
        deltaGamma = gamma_old - gamma
        assert np.shape(gamma_old) == np.shape(gamma)
        #print(gamma_old, gamma)
        #print(deltaGamma)
        dg = npla.norm(deltaGamma, 1) / np.size(deltaGamma)
        if dg < 1e-4:
            gamma_converged = True
        else:
            gamma_converged = False
        gamma_old = gamma

        # Check phi convergence
        deltaPhi = phi_old - phi
        assert np.shape(phi_old) == np.shape(phi)
        dp = npla.norm(deltaPhi,1) / np.size(deltaPhi)
        if dp < 1e-4:
            phi_converged = True
        else:
            phi_converged = False
        phi_old = phi

    return (gamma,phi)


def lda_test_driver(path, ntopics):
    wordPool = WordPool(path)
    nwords = wordPool.wordct
    ndocs = wordPool.ndocs

    calcLambdas = False

    alpha = 0.3
    eta = 0.3

    if calcLambdas:
        print("eta = " + str(eta) + " ;;; alpha = " + str(alpha))
        print("Begin lambda calculations")
        globalLambda = lda_documents(path,ntopics,alpha,eta,wordPool)

        with open("lambda.pickle",'wb') as picklefile:
            pickle.dump(globalLambda, picklefile)

    else:
        print("Load lambda from pickle")
        with open("lambda.pickle",'rb') as picklefile:
            globalLambda = pickle.load(picklefile)

    print("Calculate per-document parameters")

    nDocs = len(os.listdir(path))

    docPhi = [None] * nDocs
    docGamma = [None] * nDocs

    calcDocParams = False

    if calcDocParams:
        for j in range(nDocs):
            print(j)
            (g,p) = genDocDistributions(globalLambda, ntopics, alpha, eta, j, path, wordPool)
            docPhi.append(p)
            docGamma.append(g)

        with open("phigamma.pickle",'wb') as picklefile:
            pickle.dump((docPhi,docGamma), picklefile)
    else:
        with open('phigamma.pickle','rb') as picklefile:
            (docPhi, docGamma) = pickle.load(picklefile)

def main(args):
    path = args[0]
    topics = int(args[1])

    lda_test_driver(path,topics)

if __name__ == "__main__":
    main(sys.argv[1:])
