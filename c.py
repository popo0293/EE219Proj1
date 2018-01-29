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

classes = []
cat4 = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
for i in range(len(cat4)):
    classes.append(all_data.target_names.index(categories[i]))

for i in range(len(cat4)):
    arr = np.array(M_train_tficf.toarray()[classes[i], :])
    max_term = arr.argsort()[-10:][::-1]
    print("Top 10 terms in %s" % categories[i])
    for j in max_term:
        print(list(vectorizer.vocabulary_.keys())[j])

logging.info("finished Problem c")