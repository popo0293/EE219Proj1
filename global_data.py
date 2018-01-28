import numpy as np
import logging
from logging.config import fileConfig
from sklearn.datasets import fetch_20newsgroups

# create logger
fileConfig('logging_config.ini')
logger = logging.getLogger()

# globals

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

cat_comp = categories[:4]   # Computer Technologies
cat_rec = categories[4:]    # Recreational Activities
cat_sum_train = []          # training data size of each class 0-7
train_data = []             # training data of each class 0-7
cat_sum_test = []           # testing data size of each class 0-7
test_data = []              # testing data of each class 0-7

logging.debug("loading data")

for i in range(len(categories)):
    temp = fetch_20newsgroups(subset='train', categories=[categories[i]], shuffle=True, random_state=42)
    train_data.append(temp)
    cat_sum_train.append(len(temp.data))
    temp = fetch_20newsgroups(subset='test', categories=[categories[i]], shuffle=True, random_state=42)
    test_data.append(temp)
    cat_sum_test.append(len(temp.data))

logging.debug("loading finished")
