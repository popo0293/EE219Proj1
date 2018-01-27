import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF


class SparseToDenseArray(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, *_):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X

    def fit(self, *_):
        return self


# globals

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

cat_comp = categories[:4]   # Computer Technologies
cat_rec = categories[4:]    # Recreational Activities
cat_sum = []                # training data size of each class 0-7
train_data = []             # training data of each class 0-7

for i in range(len(categories)):
    temp = fetch_20newsgroups(subset='train', categories=[categories[i]], shuffle=True, random_state=42)
    train_data.append(temp)
    cat_sum.append(len(temp.data))


def part_a():
    y_pos = np.arange(len(categories))
    pl.figure(1)
    pl.barh(y_pos, cat_sum, align='center', color='blue', ecolor='black')
    pl.xlabel("Set Size")
    pl.ylabel("Classes")
    pl.title("Size of training set for each class")
    pl.yticks(y_pos, categories)
    pl.gca().invert_yaxis()
    pl.tight_layout()
    pl.show(block=True)


def part_a2():
    pl.figure(2)
    pl.hist(cat_sum)
    pl.title("Distribution of training size per class")
    pl.xlabel("Training size per class")
    pl.ylabel("Numbers")
    pl.show(block=True)


def part_b():
    pass


part_a()
part_a2()
