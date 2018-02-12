# K-Nearest Neighbors Implementation

This project implements the k-nearest neighbors algorithm and illustrates it in two simple examples.

### Instructions

To run, you need the Kaggle's data set on [housing prices in King County, Washington, USA](https://www.kaggle.com/harlfoxem/housesalesprediction). Place the file kc_house_data.csv in the same directory as the code. Run knn.py and then execute run_model(module) to run either of the modules. Here, module is either basic_knn or kc_housing_knn. Before the basic_knn module can be run, you need to generate the synthetic data used by that module. Do so by executing synthetic_data(). run_model() returns a data frame that shows predicted and actual labels, as well as a score.

### Explanation

K-nearest neighbors is a machine learning algorithm that predicts the outcome of a new data point by examining the k most similar examples in the training set. In the case of classification, the new prediction is taken to be the most common label of the k nearest neighbors in the training set. In the case of regressions, the predicted label is the average of the labels of the k nearest neighbors. Here "nearest" is defined by a distance function, generally Euclidean distance, though custom distance functions are possible.

### Illustration 1: Synthetic Data

The basic_knn model is a classification on a very simple data set. All training and test examples have features x and y in the range [-1,1]. The label z is 1 if the point (x,y) is in the first or third quadrant (that is, either x and y are both positive or both negative). Otherwise, z = 0. In this example, k-nearest neighbors, with k=3, should give an accuracy between 96% and 97%.

### Illustration 2: King County house prices

This example uses the above-mentioned King County housing price data set. The King County analysis uses 16 features. All inputs are normalized on the training set so that the mean is 0 and standard deviation is 1. This is necessary to insure that all features make roughly similar contributions to distance calculations. Binary features, such as whether the property is a waterfront property, are treated numerically. The distance is the standard Euclidean distance. We again take k=3.

Algorith performance is rather disappointing. The observed R2 is 0.53, inferior to the value we found earlier with [standard machine learning algorithms](https://github.com/pepper2000/basic_ml). Furthermore, the running time is quite long--over two hours on my machine.