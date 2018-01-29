from global_data import *
from timeit import default_timer as timer
from utils import *

if __name__ == "__main__":
    logging.info("Problem b")
    start = timer()
    X_train_tfidf = doTFIDF(train_data.data)
    print("With min_df = %d , (training documents, terms extracted): " % MIN_DF, X_train_tfidf.shape)
    duration = timer() - start
    logging.debug("Computation Time in secs: %d" % duration)
    logging.info("finished Problem b")
