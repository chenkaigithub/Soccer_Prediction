import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

mat = pd.read_csv("games.csv").as_matrix()

start_year = 5
year_col = mat[:,0].copy().reshape(-1,1) - start_year
encoded_sides = OneHotEncoder(n_values=[107,107], sparse=False).fit_transform(mat[:,1:3])
x = np.hstack([year_col, encoded_sides])

y = mat[:,3].copy()
y = (y > 0)*1 + (y < 0)*2

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

model = svm.LinearSVC(verbose=True, max_iter=20000, C=1)
model.fit(train_x, train_y)

print("\nsuccess rate: %f \n" % model.score(test_x, test_y))
plt.hist(model.predict(test_x))
plt.show()
