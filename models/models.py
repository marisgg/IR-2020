#!/usr/bin/env python3

import numpy as np
import json
import os
from os import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import preprocess_data

class TF_IDF:
    alpha = 0.5

    """
    Compute TF-IDF, which consists of the following two components:
    1. Term frequency: measures the frequency of a word in a document, normalize.
        tf(t,d) = count of t in d / number of words in d
    2. Inverse document frequency: measures the informativeness of term t.
        idf(t) = log(N / (df + 1)               (df = occurences of t in documents)

    The resulting formula: tf-idf(t,d) = tf(t,d)*log(N/(df+1))

    INPUT:		Dictionary, with for each file a sub-dictionary containing the title, abstract, and introduction.
    OUTPUT:		
    """ 

    def wordFrequencies(self, documentDict):
        freqDict = {}
        for docuid, document in documentDict.items():
            for _, body in document.items():
                for token in body:
                    if token not in freqDict.keys():
                        freqDict[token] = 1
                    else:
                        freqDict[token] += 1
        return freqDict

    """
    tf = (freq of word w in sentence s)/(number of words in s)
    """
    def tf(self, word, document):
        pass

    def tfDict(self, wordDictionary, numOfWords): 
        tfDictionary = {}
        for word, wordCount in wordDictionary.items():
            tfDictionary[word] = wordCount / float(numOfWords)
        return tfDictionary

    def idf(self, word, corpus):
        return np.log10()

    def tf_idf(self, word, document, corpus):
        return tf(word, document) * idf(word, corpus)

if __name__ == "__main__":
    dict = preprocess_data.preprocessing(preprocess_data.test_input)
    tf_idf = TF_IDF()
    fDict = tf_idf.wordFrequencies(dict)
    print(fDict)
    #print(json.dumps(dict, indent=4, sort_keys=True))