import string
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, TruncatedSVD
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np

'''
try:
    nltk.download("stopwords")  # if the host does not have the package
except (RuntimeError):
    pass
'''

# globals
MIN_DF = 2


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


def doTFIDF(data, mindf):
    vectorizer = CountVectorizer(min_df=mindf, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
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

'''
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
'''


def analyze(label, prob, predict):
    fpr, tpr, thresholds = roc_curve(label, prob)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='lightsteelblue',
    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='deeppink', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    cm = confusion_matrix(label, predict)
    classes = ["Computer Technology","Recreational Activity"]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.BuGn)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in [(0,0),(0,1),(1,0),(1,1)]:
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print("accuracy: ", accuracy_score(label, predict))
    print("recall: ", recall_score(label, predict))
    print("precision: ", precision_score(label, predict))
    return
