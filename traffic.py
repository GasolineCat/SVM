# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
import sklearn.metrics as sm
# load data set
filename = 'traffic_data.txt'

X = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)

X = np.array(X)

# label_encoded
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# create SVM rbf
params = {'kernel': 'rbf', 'C': 10, 'epsilon': 0.2}
regressor = SVR(**params)
regressor.fit(X, y)

# cross validation

y_pred = regressor.predict(X)
print "Mean absolute error: =", round(sm.mean_absolute_error(y, y_pred), 2)


# predict a example
input_data = ['Friday', '19:20', 'St. Louis', 'no'] # 39
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = label_encoder[count].transform(list([input_data[i]]))[0]
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

print "Predict traffic is :", int(regressor.predict(input_data_encoded)[0])