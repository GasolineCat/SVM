# -*- coding:utf-8 -*- 
# Author: Roc-J

import utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import classification_report

# load data
filename = 'data_multivar_imbalance.txt'
X, y = utilities.load_data(filename)

# classifier
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], facecolor='black', edgecolors='black', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], facecolor='None', edgecolors='black', marker='s')
plt.title('Input data')

plt.show()

# Train and test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)

# params = {'kernel': 'linear'}
params = {'kernel': 'linear', 'class_weight': 'balanced'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()

utilities.plot_classifier(classifier, X_test, y_test, 'Testing dataset')
plt.show()
y_test_pred = classifier.predict(X_test)

# measure
target_names = ['Class-' + str(int(i)) for i in set(y)]
print "#"*20
print "\n Classifier performance on training dataset\n"
print classification_report(y_train, classifier.predict(X_train), target_names=target_names)
print "#"*20

print '#'*20
print "\n Classifier performance on testing dataset\n"
print classification_report(y_test, classifier.predict(X_test), target_names=target_names)
print '#'*20