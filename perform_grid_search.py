# -*- coding:utf-8 -*- 
# Author: Roc-J

import utilities
import matplotlib.pyplot as plt
from sklearn import svm, grid_search, cross_validation
from sklearn.metrics import classification_report

'''
find the best parameters
'''
filename = 'data_multivar.txt'
X, y = utilities.load_data(filename)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, train_size=0.25, random_state=5)

parameter_grid = [
    {'kernel': ['linear'], 'C': [1, 10, 50, 600]},
    {'kernel': ['poly'], 'degree': [2, 3]},
    {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [1, 10, 50, 600]}
]

metrics = ['precision', 'recall_weighted']

for metric in metrics:
    print "\n### Searching optimal hyperparameters for ", metric

    classifier = grid_search.GridSearchCV(svm.SVC(C=1), parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    print '----- measure scores-------'
    print "\nScores across the parameter grid:"
    for params, avg_score, _ in classifier.grid_scores_:
        print params, '-->', round(avg_score, 3)

    print "\n Higtest scoring parameter set: ", classifier.best_params_

    y_pred = classifier.predict(X_test)
    print "\nFull performance report:\n"
    print classification_report(y_test, y_pred)