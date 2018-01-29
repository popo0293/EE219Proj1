import string
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, TruncatedSVD
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

'''
try:
    nltk.download("stopwords")  # if the host does not have the package
except (RuntimeError):
    pass
'''

# globals
MIN_DF = 5


class SparseToDenseArray(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, *_):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X

    def fit(self, *_):
        return self


def stem_and_tokenize(doc):
    exclude = set(string.punctuation)
    no_punctuation = ''.join(ch for ch in doc if ch not in exclude)
    tokenizer = RegexpTokenizer("[\w']+")
    tokens = tokenizer.tokenize(no_punctuation)
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    return [stemmer.stem(t) for t in tokens]

tfidf_transformer = TfidfTransformer(sublinear_tf=True, smooth_idf=False, use_idf=True)


def doTFIDF(data):
    vectorizer = CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
    m = vectorizer.fit_transform(data)
    m_train_tfidf = tfidf_transformer.fit_transform(m)
    return m_train_tfidf


def test_stem_count_vectorize():
    test_string = ["Hello, Google. But I can't answer this call go going goes bowl bowls bowled!"]
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
    X = vectorizer.fit_transform(test_string)
    feature_name = vectorizer.get_feature_names()
    print(feature_name)
    print(X.toarray())


from sklearn.pipeline import Pipeline
pipeline1 = Pipeline([
    ('vect', CountVectorizer(min_df=1, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', NMF(n_components=50, init='random', random_state=0)),
    ('clf', MultinomialNB()),
])
pipeline2 = Pipeline([
    ('vect', CountVectorizer(min_df=1, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', NMF(n_components=50, init='random', random_state=0)),
    ('toarr', SparseToDenseArray()),
    ('clf', GaussianNB()),
])

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc