from global_data import *
from utils import *

logging.debug("Problem b")

vectorizer_2 = CountVectorizer(min_df=2, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
X_2 = vectorizer_2.fit_transform(test_string)
vectorizer_5 = CountVectorizer(min_df=5, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
X_5 = vectorizer_5.fit_transform(test_string)



logging.debug("finished Problem b")
