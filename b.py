from global_data import *
from timeit import default_timer as timer
from utils import *

logging.debug("Problem b")

start = timer()
vectorizer_2 = CountVectorizer(min_df=2, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
X_2 = vectorizer_2.fit_transform(train_data.data)
X_2_train_tfidf = tfidf_transformer.fit_transform(X_2)
print("With min_df=2, (training documents, terms extracted): ", X_2_train_tfidf.shape)

vectorizer_5 = CountVectorizer(min_df=5, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
X_5 = vectorizer_5.fit_transform(train_data.data)
X_5_train_tfidf = tfidf_transformer.fit_transform(X_5)
print("With min_df=5, (training documents, terms extracted): ", X_5_train_tfidf.shape)

duration = timer()-start
print("Computation Time in secs: ", duration)

logging.debug("finished Problem b")
