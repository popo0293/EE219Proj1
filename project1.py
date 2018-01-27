import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.datasets import fetch_20newsgroups

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

cat_comp = categories[:4]
cat_rec = categories[4:]
cat_sum = []
data = []

for i in range(len(categories)):
    temp = fetch_20newsgroups(subset='all', categories=[categories[i]], shuffle=True, random_state=42)
    data.append(temp)
    cat_sum.append(len(temp.data))

print(cat_sum)

y_pos = np.arange(len(categories))
pl.figure(1)
pl.barh(y_pos, cat_sum, align='center', color='blue', ecolor='black')
pl.xlabel("Set Size")
pl.ylabel("Classes")
pl.title("Size of data set for each class")
pl.yticks(y_pos, categories)
pl.gca().invert_yaxis()
pl.tight_layout()
plt.show(block=True)
