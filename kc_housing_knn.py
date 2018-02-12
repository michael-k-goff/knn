# King County housing price data set
# Data: from Kaggle, housing prices in King County, Washington from May 2014 to May 2015
# https://www.kaggle.com/harlfoxem/housesalesprediction

import pandas as pd
import numpy as np

def normalize_year(row):
    if row["yr_renovated"] == 0:
        row["yr_renovated"] = 1900
    return row

# Read the training and test data defined above
def get_data():
    # Copying some of this from an earlier project.
    king_county_df = pd.read_csv("kc_house_data.csv")
    king_county_df = king_county_df[~king_county_df["id"].duplicated()] # Get rid of some duplicate rows
    king_county_df["bedrooms"][15870] = 3 # This is the most glaring outlier value. Making my best guess as to what it should be
    # So the nonzero years aren't so scrunched together, treat unknown renovation date as 1900
    king_county_df = king_county_df.apply(normalize_year,axis=1)
    # Get rid of columns that we won't use
    # id and date shouldn't be very relevant. sqft_living is the sum of other columsn. Using lat and long instead of zipcode for location
    king_county_df = king_county_df.drop(columns=['id', 'date','sqft_living','zipcode'])
    # Split into training and test
    np.random.seed(7777)
    rand_nums = np.random.rand(len(king_county_df))
    # This should leave about 200 test cases
    msk_train = rand_nums < 0.99
    msk_test = rand_nums >= 0.99
    df_train = king_county_df[msk_train]
    df_test = king_county_df[msk_test]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    # Normalize columns
    for colname in ["bedrooms","bathrooms","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","lat","long","sqft_living15","sqft_lot15"]:
        col_mean = np.mean(df_train[colname])
        col_std = np.std(df_train[colname])
        df_train[colname] = (df_train[colname]-col_mean)/col_std
        df_test[colname] = (df_test[colname]-col_mean)/col_std
    return df_train, df_test
    
# Distance function
def distance(p1, p2):
    # No need to take square root since we only need the relative ordering of distances
    colnames =  ["bedrooms","bathrooms","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","lat","long","sqft_living15","sqft_lot15"]
    return sum([(p1[c]-p2[c])*(p1[c]-p2[c]) for c in colnames])
    
# Scoring method: fraction of results that are right. 
def score_model(df_result):
    true_var = np.var(df_result["actual"])
    sum_square_error = sum((df_result["actual"]-df_result["pred"])*(df_result["actual"]-df_result["pred"])) / len(df_result)
    return 1 - sum_square_error/true_var
    
# Parameters needed to run the model
regression = True
response_var = "price"