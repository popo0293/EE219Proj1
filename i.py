from utils import *
from global_data import *

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

logging.info("Problem i")

method_arr = [TruncatedSVD(n_components=50, n_iter=10, random_state=17),
              NMF(n_components=50, random_state=17)]
method_name = ["LSI", "NMF"]
norm_arr = ["l1", "l2"]

for ai, method in enumerate(method_arr):
    for bi, nm in enumerate(norm_arr):
        pipeline_h = Pipeline([
            ('vect', CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)),
            ('tfidf', TfidfTransformer()),
            ('reduce_dim', method),
            ('clf', LogisticRegression(penalty=nm)),
        ])
        pipeline_h.fit(train_data.data, train_label)
        pred_test = pipeline_h.predict(test_data.data)
        pred_test_prob = pipeline_h.predict_proba(test_data.data)[:, 1]
        print("-" * 70)
        print("Using " + method_name[ai] +
              " and Logistic Regression with penalty: " + nm)
        analyze(test_label, pred_test_prob, pred_test, CAT, 2)

logging.info("finished Problem i")
