from utils import *
from global_data import *

method_arr = [TruncatedSVD(n_components=50, n_iter=10, random_state=17),
              NMF(n_components=50, random_state=17)]
method_name = ["LSI", "NMF"]

for ai, method in enumerate(method_arr):
    pipeline_e = Pipeline([
        ('vect', CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)),
        ('tfidf', TfidfTransformer()),
        ('reduce_dim', method),
        ('clf', GaussianNB()),
    ])
    pipeline_e.fit(train_data.data, train_label)
    pred_test = pipeline_e.predict(test_data.data)
    pred_test_prob = pipeline_e.decision_function(test_data.data)
    print("-" * 70)
    print("Using min_df=2 (fixed from now on), " + method_name[ai] +
          " ,and gamma = %f" % g)
    analyze(test_label, pred_test_prob, pred_test)
