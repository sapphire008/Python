# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 01:41:38 2015

@author: Edward
"""
import numpy as np

import matplotlib.pyplot as plt

# line plot
# initialize some data
X = np.arange(0, 10, 0.1)
Y1 = np.sin(X)
Y2 = np.cos(X)
fig = plt.figure(5, figsize=(8,3))
plt.plot(X,Y1, label="Sin")
plt.plot(X,Y2, label="Cos")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.legend()

# Scatter plot
from sklearn import datasets
# Load iris dataset
iris = datasets.load_iris()

print(iris.feature_names)
# initialize a figure
fig = plt.figure(1, figsize=(4,3))
# Scatter plot
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
# Set x/y labels
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])


# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
# initialize a figure
fig = plt.figure(2, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(iris.data[:,0], iris.data[:,1], iris.data[:,2], c=iris.target)
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])
plt.show()

# Histogram
fig = plt.figure(3, figsize=(4,3))
plt.hist(iris.data[:,2], bins=7, alpha=0.4, hatch="/")
plt.xlabel(iris.feature_names[2])
plt.ylabel('Count')
# grab figure axs for manipulations
ax = fig.axes[0]
# Set styles of axes
ax.tick_params(axis='both',direction='out')
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Barplot
print(iris.target_names)
# First, count each target
target, count = np.unique(iris.target, return_counts=True)
target = target+0.25
fig = plt.figure(4, figsize=(4,3))
plt.bar(target, count, width=0.60, yerr=[5, 8, 10])
plt.xticks(target+0.30, iris.target_names)
ax = fig.axes[0]
ax.tick_params(axis='both',direction='out')
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('left')



########### Change font ####################
# initialize a figure
fig = plt.figure(1, figsize=(4,3))
# Scatter plot
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
# Set x/y labels
plt.xlabel(iris.feature_names[0], family='Arial', fontsize=12)
plt.ylabel(iris.feature_names[1], family='Arial', fontsize=12)
# Set font
import matplotlib.font_manager as fm
ax = fig.axes[0]
fontprop = fm.FontProperties(family='Times New Roman', style="normal", size=12)
ax.xaxis.label.set_fontproperties(fontprop)
ax.yaxis.label.set_fontproperties(fontprop)
# Set font of tick labels
[a.set_fontproperties(fontprop) for a in ax.get_xticklabels()]
[a.set_fontproperties(fontprop) for a in ax.get_yticklabels()]




