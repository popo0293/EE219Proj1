from utils import *
from global_data import *
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

logging.info("Problem f")

# create labels
# 0 for computer technology, 1 for recreational activities
train_label = [(x//4) for x in train_data.target]
test_label = [(x//4) for x in test_data.target]

gamma = [1000, 100, 10, 1, 0.1, 0.01, 0.001]

method_arr = [TruncatedSVD(n_components=50, n_iter=10, random_state=17),
              NMF(n_components=50, random_state=17)]
method_name = ["LSI", "NMF"]

train_reduced = []
test_reduced = []

for method in method_arr:
    pipeline_f = Pipeline([
        ('vect', CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)),
        ('tfidf', TfidfTransformer()),
        ('reduce_dim', method),
    ])
    train_reduced.append(pipeline_f.fit_transform(train_data.data))
    test_reduced.append(pipeline_f.fit_transform(test_data.data))


best_g = []
for ai, method in enumerate(method_name):
    print("Using method "+method)
    Score = []
    for g in gamma:
        Score.append(np.average(cross_val_score(LinearSVC(C=g), train_reduced[ai], train_label, cv=5, n_jobs=-1)))
    plt.figure()
    plt.xlabel("Gamma")
    plt.ylabel('5-fold Cross Validation Score')
    plt.ylim([0.0, 1.05])
    plt.xscale('log')
    plt.plot(gamma, Score)
    plt.show()
    best_g.append(gamma[np.argmax(Score)])
    print("Best value for gamma: ", best_g[ai])

    pipeline_f = Pipeline([
        ('vect', CountVectorizer(min_df=mdf, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)),
        ('tfidf', TfidfTransformer()),
        ('reduce_dim', TruncatedSVD(n_components=50, n_iter=10, random_state=17)),
        ('clf', LinearSVC(C=best_g[ai])),
    ])
    pipeline_f.fit(train_data.data, train_label)
    pred_test = pipeline_f.predict(test_data.data)
    pred_test_prob = pipeline_f.decision_function(test_data.data)
    print("-" * 70)
    print("Using method "+method+" and best gamma %f" % best_g[ai])
    analyze(test_label, pred_test_prob, pred_test)
    print("-" * 70)


logging.info("finished Problem e")
