import numpy as np
import pylab as pl
import global_data as gd

y_pos = np.arange(len(gd.categories))
pl.figure(1)
pl.barh(y_pos, gd.cat_sum_train, align='center', color='blue', ecolor='black')
pl.xlabel("Set Size")
pl.ylabel("Classes")
pl.title("Size of training set for each class")
pl.yticks(y_pos, gd.categories)
pl.gca().invert_yaxis()
pl.tight_layout()

pl.figure(2)
pl.hist(gd.cat_sum_train)
pl.title("Distribution of training size per class")
pl.xlabel("Training size per class")
pl.ylabel("Numbers")
pl.show()
