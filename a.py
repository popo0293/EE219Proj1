import numpy as np
import pylab as pl
from global_data import *

logging.debug("Problem a")

cat_sum_train = []          # training data size of each class 0-7
cat_sum_test = []           # testing data size of each class 0-7

for i in range(len(categories)):
    temp = fetch_20newsgroups(subset='train', categories=[categories[i]], shuffle=True, random_state=42)
    cat_sum_train.append(len(temp.data))
    temp = fetch_20newsgroups(subset='test', categories=[categories[i]], shuffle=True, random_state=42)
    cat_sum_test.append(len(temp.data))

y_pos = np.arange(len(categories))
pl.figure(1)
pl.barh(y_pos, cat_sum_train, align='center', color='blue', ecolor='black')
pl.xlabel("Set Size")
pl.ylabel("Classes")
pl.title("Size of training set for each class")
pl.yticks(y_pos, categories)
pl.gca().invert_yaxis()
pl.tight_layout()

pl.figure(2)
pl.hist(cat_sum_train)
pl.title("Distribution of training size per class")
pl.xlabel("Training size per class")
pl.ylabel("Numbers")
pl.show()

sum = 0
for i in cat_sum_train:
    sum += i

print(sum)

logging.debug("finished Problem a")
