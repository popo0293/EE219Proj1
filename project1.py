import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

cat_comp = categories[:4]
cat_rec = categories[4:]
cat_sum = []
data = []

for i in range(len(categories)):
    temp = fetch_20newsgroups(categories=[categories[i]], shuffle=True, random_state=42)
    data.append(temp)
    cat_sum.append(len(temp.data))

print(cat_sum)

plt.figure(1)
plt.title("Size of data set for each class")
plt.xlabel(categories)
plt.ylabel("Numbers")

plt.show(block=True)
