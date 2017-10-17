# -*- coding:utf-8 -*- 
# Author: Roc-J

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import utilities

filename = sys.argv[1]
X, y = utilities.load_data(filename)

class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], facecolor='black', edgecolors='black', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], facecolor='None', edgecolors='black', marker='s')
plt.title('Input data')
plt.show()

# Train test split and SVM training
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)

# create the classifier
# params = {'kernel':'linear'}
# params = {'kernel': 'poly', 'degree':3}
params = {'kernel': 'rbf'}
classifier = SVC(**params)

classifier.fit(X_train, y_train)

# plot
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()

# predict
y_test_predict = classifier.predict(X_test)
utilities.plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()

target_names = ['Class-' + str(int(i)) for i in set(y)]
print "\n" + '#'*20
print "\nClassifier performance on training dataset\n"
print classification_report(y_train, classifier.predict(X_train), target_names=target_names)
print "#"*30 + "\n"

print "#"*20
print "\nClassification report on test dataset\n"
print classification_report(y_test, y_test_predict, target_names=target_names)
print '#'*20