from global_data import *
from timeit import default_timer as timer
from utils import *

logging.debug("Problem c")

start = timer()

all_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

string_of_each_class = []
for i in range(len(all_data.target_names)):
    string_of_each_class.append("")

for i in range(len(all_data.data)):
    string_of_each_class[all_data.target[i]] += " " + all_data.data[i]

vectorizer_2 = CountVectorizer(min_df=2, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
X_2 = vectorizer_2.fit_transform(string_of_each_class)
X_2_train_tfidf = tfidf_transformer.fit_transform(X_2)
print("With min_df=2, (training documents, terms extracted): ", X_2_train_tfidf.shape)

vectorizer_5 = CountVectorizer(min_df=5, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
X_5 = vectorizer_5.fit_transform(string_of_each_class)
X_5_train_tfidf = tfidf_transformer.fit_transform(X_5)
print("With min_df=5, (training documents, terms extracted): ", X_5_train_tfidf.shape)

duration = timer()-start
print("Computation Time in secs: ", duration)

logging.debug("finished Problem c")