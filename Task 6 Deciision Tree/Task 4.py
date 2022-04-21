# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 11:05:05 2021

@author: sRv
"""
# Importing libraries in Python
import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree

# Loading the iris dataset
iris=datasets.load_iris()

# Defining the decision tree algorithm
dtree = DecisionTreeRegressor()
dtree_model = dtree.fit(iris.data, iris.target)

# Data Visualisation
plt.figure(figsize = (32, 14))
tree.plot_tree(dtree_model)
plt.show()


