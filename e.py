from utils import *
from global_data import *

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logging.info("Problem e")

# compare the effect of min_df with LSI only
min_df_arr = [MIN_DF, 5]
gamma = [1000, 0.001]
for ai, mdf in enumerate(min_df_arr):
    for bi, g in enumerate(gamma):
        pipeline_lsi = Pipeline([
            ('vect', CountVectorizer(min_df=mdf, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)),
            ('tfidf', TfidfTransformer()),
            ('reduce_dim', TruncatedSVD(n_components=50, n_iter=10, random_state=17)),
            ('clf', LinearSVC(C=g)),
        ])
        pipeline_lsi.fit(train_data.data, train_label)
        pred_test = pipeline_lsi.predict(test_data.data)
        pred_test_prob = pipeline_lsi.decision_function(test_data.data)
        print("-" * 70)
        print("Using Min_df = %d" % mdf," and gamma = %f" % g)
        analyze(test_label, pred_test_prob, pred_test, CAT, 2)

print("Now compare NMF and LSI")
method_arr = [TruncatedSVD(n_components=50, n_iter=10, random_state=17),
              NMF(n_components=50, random_state=17)]
method_name = ["LSI", "NMF"]

for ai, method in enumerate(method_arr):
    for bi, g in enumerate(gamma):
        pipeline_e = Pipeline([
            ('vect', CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)),
            ('tfidf', TfidfTransformer()),
            ('reduce_dim', method),
            ('clf', LinearSVC(C=g)),
        ])
        pipeline_e.fit(train_data.data, train_label)
        pred_test = pipeline_e.predict(test_data.data)
        pred_test_prob = pipeline_e.decision_function(test_data.data)
        print("-" * 70)
        print("Using min_df=2 (fixed from now on), " + method_name[ai] +
              " ,and gamma = %f" % g)
        analyze(test_label, pred_test_prob, pred_test, CAT, 2)

logging.info("finished Problem e")
