from global_data import *
from timeit import default_timer as timer
from utils import *

logging.info("Problem c")

start = timer()

all_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

string_of_each_class = []
for i in range(len(all_data.target_names)):
    string_of_each_class.append("")

for i in range(len(all_data.data)):
    string_of_each_class[all_data.target[i]] += " " + all_data.data[i]

vectorizer = CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
M = vectorizer.fit_transform(string_of_each_class)
M_train_tficf = tfidf_transformer.fit_transform(M)
print("With min_df = %d, (training documents, terms extracted): " % MIN_DF, M_train_tficf.shape)

duration = timer()-start
print("Computation Time in secs: ", duration)

cat_top10 = ["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "misc.forsale", "soc.religion.christian"]
print("With min_df = %d" % MIN_DF)
for name in cat_top10:
    index = all_data.target_names.index(name)
    arr = M_train_tficf.toarray()[index, 1:10]
    sig_terms = np.argsort(arr)[-10:][-1::-1]
    print("Top 10 most significant terms in class %s are:" % name)
    for i in range(10):
        print(list(vectorizer.vocabulary_.keys())[i])

logging.info("finished Problem c")
