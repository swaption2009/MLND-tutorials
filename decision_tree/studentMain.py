#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)


#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())


### measure accuracy score
from sklearn.metrics import accuracy_score

pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "accuracy score: ", acc

def submitAccuracies():
    return {"acc": round(acc, 3)}



### Decision tree accuracy with min_samples_split 2 and 50
from sklearn import tree

clf2 = tree.DecisionTreeClassifier(min_samples_split = 2)
clf2 = clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
acc_min_samples_split_2 = accuracy_score(pred2, labels_test)

clf50 = tree.DecisionTreeClassifier(min_samples_split = 50)
clf50 = clf50.fit(features_train, labels_train)
pred50 = clf50.predict(features_test)
acc_min_samples_split_50 = accuracy_score(pred50, labels_test)

print "Accuracy with min_samples_split = 2 is", acc_min_samples_split_2
print "Accuracy with min_samples_split = 50 is", acc_min_samples_split_50

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}

