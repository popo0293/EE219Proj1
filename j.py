from utils import *
from global_data import *

from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC

logging.info("Problem j")

cat_top10 = ["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "misc.forsale", "soc.religion.christian"]

train_mult_data = fetch_20newsgroups(subset='train', categories=cat_top10, shuffle=True, random_state=42)
test_mult_data = fetch_20newsgroups(subset='test', categories=cat_top10, shuffle=True, random_state=42)

method_arr = [TruncatedSVD(n_components=50, n_iter=10, random_state=17),
              NMF(n_components=50, random_state=17)]
method_name = ["LSI", "NMF"]
clf_arr = [GaussianNB(),
           OneVsOneClassifier(LinearSVC()),
           OneVsRestClassifier(LinearSVC())]
clf_name = ["GaussianNaiveBayes", "OneVsOne(svm)", "OneVsRest(svm)"]

for ai, method in enumerate(method_arr):
    for bi, clf in enumerate(clf_arr):
        pipeline_j = Pipeline([
            ('vect', CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)),
            ('tfidf', TfidfTransformer()),
            ('reduce_dim', method),
            ('clf', clf),
        ])
        pipeline_j.fit(train_mult_data.data, train_mult_data.target)
        pred_test = pipeline_j.predict(test_mult_data.data)
        pred_test_prob = pipeline_j.predict_proba(test_data.data)[:, 1]
        print("-" * 70)
        print("Using " + method_name[ai] +
              " and Learning algorithm: " + clf_name[bi])
        analyze(test_mult_data.target, pred_test_prob, pred_test, cat_top10, len(cat_top10))

logging.info("finished Problem j")
