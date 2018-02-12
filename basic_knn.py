# Very basic kNN

import random
import pandas as pd

# Generate a synthetic data set to test kNN
# This must be run separately before running the model
def synthetic_data():
    num_train = 1000
    num_test = 200
    
    x_train = [random.random()*2-1.0 for i in range(num_train)]
    y_train = [random.random()*2-1.0 for i in range(num_train)]
    z_train = [int(x_train[i]*y_train[i]>0) for i in range(num_train)]
    df_train = pd.DataFrame(data={"x":x_train,"y":y_train,"z":z_train})
    df_train.to_csv("train_data.csv",index=False)
    
    x_test = [random.random()*2-1.0 for i in range(num_test)]
    y_test = [random.random()*2-1.0 for i in range(num_test)]
    z_test = [int(x_test[i]*y_test[i]>0) for i in range(num_test)]
    df_test = pd.DataFrame(data={"x":x_test,"y":y_test,"z":z_test})
    df_test.to_csv("test_data.csv",index=False)
    
# Read the training and test data defined above
def get_data():
    df_train = pd.read_csv("train_data.csv")
    df_test = pd.read_csv("test_data.csv")
    return df_train, df_test
    
# Distance function
def distance(p1, p2):
    # No need to take square root since we only need the relative ordering of distances
    return (p1["x"]-p2["x"])*(p1["x"]-p2["x"]) + (p1["y"]-p2["y"])*(p1["y"]-p2["y"])
   
# Scoring method: fraction of results that are right. 
def score_model(df_result):
    return 1 - sum((df_result["actual"]-df_result["pred"])*(df_result["actual"]-df_result["pred"]) > 0.1) / float(len(df_result))
    
# Parameters needed to run the model
regression = False
response_var = "z"