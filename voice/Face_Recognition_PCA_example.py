# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
from sklearn import cross_validation, datasets, decomposition, svm

# load data
lfw_people = datasets.fetch_lfw_people(resize=0.4)
perm = np.random.permutation(lfw_people.target.size)
lfw_people.data = lfw_people.data[perm]
lfw_people.target = lfw_people.target[perm]
faces = np.reshape(lfw_people.data, (lfw_people.target.shape[0], -1))
train, test = iter(cross_validation.StratifiedKFold(lfw_people.target,k=4)).next()
x_train, x_test = faces[train], faces[test]
y_train, y_test = lfw_people.target[train], lfw_people.target[test]

# dimension reduction
pca = decomposition.RandomizedPCA(n_components=150,whiten=True)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# classification
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(x_train_pca, y_train)

# predict on new images
for i in range(10):
    print lfw_people.target_names[clf.predict(x_test_pca[i])[0]]
    _=pl.imshow(x_test[i].reshape(50,37), cmap=pl.cm.gray)
    _=raw_input()


datasets.fetch_lfw_people()