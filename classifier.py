from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


main_corpus = []
main_corpus_target = []


my_categories = ["1", "2", "3", "4", "5"]

# feeding corpus the testing data

with open('hsk/1.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(1)

with open('hsk/2.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(2)

with open('hsk/3.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(3)

with open('hsk/4.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(4)

with open('hsk/5.txt') as f:
    for line in f:
        main_corpus.append(line)
        main_corpus_target.append(5)

# ratio = 25  # training to test set

import jieba

def tokenize(text):
    tokens = jieba.cut(text, cut_all=False)
    return list(tokens)


vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, max_df=0.5, tokenizer=tokenize, use_idf=True, smooth_idf=True)
analyze = vectorizer.build_analyzer()
X_train = vectorizer.fit_transform(main_corpus)


clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])

clf.fit(X_train, main_corpus_target)

from sklearn.externals import joblib
joblib.dump(clf, 'classifer.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')




