from utils import *
from global_data import *
from timeit import default_timer as timer

logging.info("Problem d")
start = timer()

X_train_tfidf = doTFIDF(train_data.data)
X_test_tfidf = doTFIDF(test_data.data)

logging.info("SVD")
svd = TruncatedSVD(n_components=50, n_iter=10, random_state=17)
X_train_tfidf_LSI = svd.fit_transform(X_train_tfidf)
X_test_tfidf_LSI = svd.fit_transform(X_test_tfidf)

logging.info("NMF")
nmf = NMF(n_components=50, random_state=17)
X_train_tfidf_NMF= nmf.fit_transform(X_train_tfidf)
X_test_tfidf_NMF = nmf.fit_transform(X_test_tfidf)

duration = timer()-start
logging.debug("Computation Time in secs: %d" % duration)

print("Min_df = %d" % MIN_DF)

logging.info("finished Problem d")
