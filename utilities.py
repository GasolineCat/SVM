# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    X = []
    y = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = [float(i) for i in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1])

    X = np.array(X)
    y = np.array(y)

    return X, y

# plot the classifier boundaries on input data
def plot_classifier(classifier, X, y, title='Classifier boundaries', annotate=False):
    # define range to plot
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    mesh_out = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_out = mesh_out.reshape(x_values.shape)

    plt.figure()
    plt.title(title)
    plt.pcolormesh(x_values, y_values, mesh_out, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.xticks()
    plt.yticks()

    if annotate:
        for x, y in zip(X[:, 0], X[:, 1]):
            # Full documentation of the function available here:
            # http://matplotlib.org/api/text_api.html#matplotlib.text.Annotation
            plt.annotate(
                '(' + str(round(x, 1)) + ',' + str(round(y, 1)) + ')',
                xy = (x, y), xytext = (-15, 15),
                textcoords = 'offset points',
                horizontalalignment = 'right',
                verticalalignment = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.6', fc = 'white', alpha = 0.8),
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))