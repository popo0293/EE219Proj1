from global_data import *
from timeit import default_timer as timer
from utils import *

logging.info("Problem c")

start = timer()

logging.info("loading training data")
all_class_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

logging.info("concatenating data")
string_of_each_class = []
for i in range(len(all_class_data.target_names)):
    string_of_each_class.append("")

for i in range(len(all_class_data.data)):
    string_of_each_class[all_class_data.target[i]] = string_of_each_class[all_class_data.target[i]]\
                                                     + " " + all_class_data.data[i]

logging.info("vectorizing")
# tfidf_vectorizer = TfidfVectorizer(min_df=MIN_DF, analyzer="word", stop_words=ENGLISH_STOP_WORDS,
#                                   sublinear_tf=True, use_idf=True,
#                                   tokenizer=stem_and_tokenize)

vectorizer = CountVectorizer(min_df=MIN_DF, analyzer="word", stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
M = vectorizer.fit_transform(string_of_each_class)
M_train_tficf = tfidf_transformer.fit_transform(M)
#M_train_tficf = tfidf_vectorizer.fit_transform(string_of_each_class)
print("With min_df = %d, (all classes, terms extracted): " % MIN_DF, M_train_tficf.shape)

duration = timer()-start
logging.debug("Computation Time in secs: ", duration)

''' debugging block
arra = M.toarray()[all_data.target_names.index("soc.religion.christian"), :]
chrisid = vectorizer.get_feature_names().index('christ')
print("value for christ is %d: " % arra[chrisid])
arrb = M_train_tficf.toarray()[all_data.target_names.index("soc.religion.christian"), :]
print("value for christ is %f: " % arrb[chrisid])
'''

cat_top10 = ["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "misc.forsale", "soc.religion.christian"]
print("With min_df = %d" % MIN_DF)

for name in cat_top10:
    index = all_class_data.target_names.index(name)
    arr = M_train_tficf.toarray()[index]
    sig_terms = np.argsort(arr)[-10:][-1::-1]
    logging.debug(name)
    logging.debug(sig_terms)
    print("Top 10 most significant terms in class %s are:" % name)
    for i in sig_terms:
        print(vectorizer.get_feature_names()[i])

    '''  debugging block
    print(arr[sig_terms])
    chrisid = vectorizer.get_feature_names().index('mac')
    print(chrisid)
    print("value for christ is %f: " % arr[chrisid])
    '''

logging.info("finished Problem c")
