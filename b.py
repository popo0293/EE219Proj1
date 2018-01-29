from global_data import *
from timeit import default_timer as timer
from utils import *

logging.info("Problem b")

start = timer()
vectorizer = CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
M = vectorizer.fit_transform(train_data.data)
M_train_tfidf = tfidf_transformer.fit_transform(M)
print("With min_df = %d , (training documents, terms extracted): " % MIN_DF, M_train_tfidf.shape)

duration = timer()-start
print("Computation Time in secs: ", duration)

logging.info("finished Problem b")
