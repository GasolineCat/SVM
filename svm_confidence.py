# -*- coding:utf-8 -*- 
# Author: Roc-J

import utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import cross_validation


filename = 'data_multivar.txt'
X, y = utilities.load_data(filename)

print u'-----svm--------'
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, train_size=0.25, random_state=5)

# params = {'kernel': 'rbf'}
params = {'kernel': 'rbf', 'probability': True}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()

########################
# measure distance from the boundary
input_datapoints = np.array([[2, 1.5], [8, 9], [4.8, 5.2], [4, 4], [2.5, 7], [7.6, 2], [5.4, 5.9]])
print "\n Distance from the boundary"
for i in input_datapoints:
    # print i, '-->', classifier.decision_function(i.reshape(1, -1))[0]
    print i, '-->', classifier.predict_proba(i.reshape(1, -1))[0]

utilities.plot_classifier(classifier, input_datapoints, [0]*len(input_datapoints), 'Input datapoints', True)
plt.show()