# -*- coding:utf-8 -*- 
# Author: Roc-J

from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing, cross_validation

# load data
# filename = 'building_event_binary.txt'
filename = 'building_event_multiclass.txt'
X = []

with open(filename, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append([data[0]] + data[2:])

X = np.array(X)

# label_encode
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
params = {'kernel': 'rbf', 'probability': True, 'class_weight': 'balanced'}
classifier = SVC(**params)
classifier.fit(X, y)

# cross_validation

accuracy = cross_validation.cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print "Accuracy of the classifier: ", round(100.0* accuracy.mean(), 2), '%'

# predict a example
# input_data = ['Wednesday', '10:00:00', '5', '8']
input_data = ['Tuesday', '13:00:00', '12', '12']
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = label_encoder[count].transform(list([input_data[i]]))[0]
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

output_class = classifier.predict(input_data_encoded)
print "output class :", label_encoder[-1].inverse_transform(output_class)[0]
