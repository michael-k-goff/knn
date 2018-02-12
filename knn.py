# k-Nearest Neighbors

# Here's a tutorial for guidance: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# Project goals
# 1) Program kNN, similar to the tutorial. Aim to make it efficient
# 2) Apply it to a synthetic data set
# 3) Come up with a pathological example that is foiled by Euclidean distance
# 4) Set up a train/dev/test data set and use it to learn k and/or a custom distance function. See this paper: https://brage.bibsys.no/xmlui/bitstream/handle/11250/253158/621995_FULLTEXT01.pdf?sequence=1&isAllowed=y
# 5) See how the sci-kit-learn kNN performs
# 6) Try kNN on the King County housing price data set

import pandas as pd
import os
import numpy as np

import basic_knn
reload(basic_knn)
import kc_housing_knn
reload(kc_housing_knn)

os.chdir("/Users/michaelgoff/Desktop/Machine Learning/Project 4 - k-Nearest-Neighbors")
    
# Find the k nearest neighbors to the point given by row, within the data set train
# The return object is a list of k triplets. Each triplet consists of the index, distance, and response value.
def find_nearest_neighbors(row,train,k,dist_func,response_var):
    indices = []
    for i in range(len(train)):
        d = dist_func(row,train.ix[i])
        if len(indices) < k:
            indices.append([i,d,train.ix[i][response_var]])
            indices = sorted(indices, key=lambda i: i[1])
        else:
            if d < indices[k-1][1]:
                indices[k-1] = [i,d,train.ix[i][response_var]]
                indices = sorted(indices, key=lambda i: i[1])
    return indices
    
# Given a set of indices and a training data set, predict the response variable of the point that is near them
# Might turn this into a more complex function, such as by weighting by distance
# The regression parameter indicates whether we are doing a regression or classification. If classifying, choose the most common of the nearest neighbor.
# If a regression, take the mean of all values. Again, modification is possible.
def predict(indices,module):
    # If regressing, take the mean of all values
    if module.regression:
        return np.mean([indices[i][2] for i in range(len(indices))])
    # Build a dictionary counting classes. Agnostic to what the output classes are
    classes = {}
    for i in range(len(indices)):
        c = indices[i][2]
        if c not in classes:
            classes[c] = 1
        else:
            classes[c] += 1
    # Return the most common classes
    frequency = {classes[i]:i for i in classes}
    return frequency[max(frequency)]
        
# Put the functions together and determine how kNN performs on the synthetic data
# Usage: pass in a module (e.g. basic_knn). It must have functions distance(), get_data(), and score_model()
def run_model(module):
    df_train, df_test = module.get_data()
    predictions = []
    for i in range(len(df_test)):
        print "Predicting " + str(i) + " of " +str(len(df_test)) + "."
        row = df_test.ix[i]
        indices = find_nearest_neighbors(row, df_train, 3, module.distance,module.response_var)
        predictions.append(predict(indices,module))
    df_result = pd.DataFrame(data = {"pred":predictions,"actual":df_test[module.response_var]})
    
    return df_result, module.score_model(df_result)
