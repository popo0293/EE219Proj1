import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

nltk.download("stopwords")


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


tfidf_transformer = TfidfTransformer()


'''
test_string = ["Hello, Google. But I can't answer this call go going goes bowl bowls bowled!"]
vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
X = vectorizer.fit_transform(test_string)
feature_name = vectorizer.get_feature_names()

print(feature_name)
print(X.toarray())

'''