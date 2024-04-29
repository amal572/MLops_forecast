import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from prefect import flow, task
from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

from sklearn import preprocessing

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,\
  mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV
from xgboost import plot_importance, plot_tree
from xgboost import XGBRegressor
import xgboost as xg
from pandas import concat
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR


def out_std(df, column):
    global lower,upper
    # calculate the mean and standard deviation of the data frame
    data_mean, data_std = df[column].mean(), df[column].std()
    # calculate the cutoff value
    cut_off = data_std * 3
    # calculate the lower and upper bound value
    lower, upper = data_mean - cut_off, data_mean + cut_off
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] > upper]
    df2 = df[df[column] < lower]
    return print('Total number of outliers are', df1.shape[0]+ df2.shape[0])

def convert(dt):
    try:
        return datetime.strptime(dt, '%d/%m/%Y').strftime('%d/%m/%Y')
    except ValueError:
        return datetime.strptime(dt, '%m/%d/%Y').strftime('%d/%m/%Y')

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def save_data(test):
    test.to_csv('test.csv')

@task(retries=3, retry_delay_seconds=2)
def read_data(filename):
    """Read data into DataFrame"""
    df = pd.read_csv(filename)
    return df


@task
def convert_date(df):
    df['date'] = df['APPLE_WEEK_DATE'].apply(convert)
    df.index = df.date
    return df


@task
def feature_encoding(df):
    le = preprocessing.LabelEncoder()
    df['COMPANY_KEY'] = le.fit_transform(df['COMPANY_KEY'])
    df['COMPANY_ITEM_KEY'] = le.fit_transform(df['COMPANY_ITEM_KEY'])
    df['BILLING_COUNTRY_KEY'] = le.fit_transform(df['BILLING_COUNTRY_KEY'])
    return df


@task
def clean_data(df):
    # Remove null value
    df =df.dropna()

    #Remove duplicate
    df = df.drop_duplicates()

    #Remove Outlier
    out_std(df,'SELL_OUT_QUANTITY')
    df = df[(df['SELL_OUT_QUANTITY'] < upper)]
    return df

@task
def feature_extraction(df):
    #df['month'] = df.index.month
    #df['day'] = df.index.day
    #df['hour'] = df.index.hour
    #df['day_of_week'] = df.index.day_of_week
    #df['day_name'] = df.index.day_name()
    #df['year'] = df.index.year
    load_val = df[['SELL_OUT_QUANTITY']]
    window = load_val.expanding()
    df['SALES_MIN_WINDOW'] = window.min()
    df['SALES_MAX_WINDOW'] = window.max()
    df['SALES_MEAN_WINDOW'] = window.mean()
    df['SALES_last_WINDOW'] = load_val. shift(-1)
    df['lag1'] = df['SELL_OUT_QUANTITY']. shift(4)
    df['lag2'] = df['SELL_OUT_QUANTITY']. shift(8)
    df['lag3'] = df['SELL_OUT_QUANTITY']. shift(12)
    df['lag4'] = df['SELL_OUT_QUANTITY']. shift(16)
    df['lag5'] = df['SELL_OUT_QUANTITY']. shift(20)
    df['lag6'] = df['SELL_OUT_QUANTITY']. shift(24)
    df['lag7'] = df['SELL_OUT_QUANTITY']. shift(28)
    df['lag8'] = df['SELL_OUT_QUANTITY']. shift(32)
    df['lag9'] = df['SELL_OUT_QUANTITY']. shift(36)
    df['lag10'] = df['SELL_OUT_QUANTITY']. shift(40)
    df['lag11'] = df['SELL_OUT_QUANTITY']. shift(44)
    df['lag12'] = df['SELL_OUT_QUANTITY']. shift(48)
    return df

@task 
def fill_null_value(df):
    df['lag1'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag2'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag3'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag4'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag5'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag6'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag7'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag8'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag9'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag10'] = df['SELL_OUT_QUANTITY'].fillna(0)
    df['lag11'] = df['SELL_OUT_QUANTITY']. fillna(0)
    df['lag12'] = df['SELL_OUT_QUANTITY']. fillna(0)
    return df

@task
def split_data(df):
    training_mask = df["date"] < "2023-01-01"
    training_data = df.loc[training_mask]

    testing_mask = df["date"] >= "2023-01-01"
    testing_data = df.loc[testing_mask]

    train = training_data.loc[training_data['COMPANY_KEY']==4]
    test = testing_data.loc[testing_data['COMPANY_KEY']==4]
    
    n = len(training_data)

    train_df = training_data[0:int(n*0.7)]
    dv = DictVectorizer()
    train_dicts = train_df.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_df = training_data[int(n*0.7):int(n*0.9)]

    val_dicts = val_df.to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    X_train = train_df[['COMPANY_KEY','COMPANY_ITEM_KEY','BILLING_COUNTRY_KEY','APPLE_YEAR','APPLE_WEEK','STOCK_QUANTITY','INVOICED_QUANTITY','IN_TRANSIT_QUANTITY'
   ,'SALES_MIN_WINDOW','SALES_MAX_WINDOW','SALES_MEAN_WINDOW','SALES_last_WINDOW','lag1','lag2'
    ,'lag3','lag4','lag5','lag6','lag7','lag8','lag9','lag10','lag11','lag12']]
    y_train = train_df["SELL_OUT_QUANTITY"]

    X_valid = val_df[['COMPANY_KEY','COMPANY_ITEM_KEY','BILLING_COUNTRY_KEY','APPLE_YEAR','APPLE_WEEK','STOCK_QUANTITY','INVOICED_QUANTITY','IN_TRANSIT_QUANTITY'
   ,'SALES_MIN_WINDOW','SALES_MAX_WINDOW','SALES_MEAN_WINDOW','SALES_last_WINDOW','lag1','lag2'
    ,'lag3','lag4','lag5','lag6','lag7','lag8','lag9','lag10','lag11','lag12']]
    y_valid = val_df["SELL_OUT_QUANTITY"]

    X_test = test[['COMPANY_KEY','COMPANY_ITEM_KEY','BILLING_COUNTRY_KEY','APPLE_YEAR','APPLE_WEEK','STOCK_QUANTITY','INVOICED_QUANTITY','IN_TRANSIT_QUANTITY'
    ,'SALES_MIN_WINDOW','SALES_MAX_WINDOW','SALES_MEAN_WINDOW','SALES_last_WINDOW','lag1','lag2'
    ,'lag3','lag4','lag5','lag6','lag7','lag8','lag9','lag10','lag11','lag12']]
    y_test = test["SELL_OUT_QUANTITY"]
    return X_train,y_train,X_valid,y_valid,X_test,y_test,test, dv



#@task(log_prints=True)
#def train_best_model(X_train,X_val,y_train,y_val):
#    """train a model with best hyperparams and write everything out"""

#    print(X_train.shape)
#    print(X_val.shape)
#    print(y_train.shape)
#    print(y_val.shape)
#    with mlflow.start_run():
#        train = xgb.DMatrix(X_train, label=y_train)
#        c
        #len(train)
        #len(valid)

#        best_params = {
#            "learning_rate": 0.09585355369315604,
#            "max_depth": 30,
#            "min_child_weight": 1.060597050922164,
#            "objective": "reg:linear",
#            "reg_alpha": 0.018060244040060163,
#            "reg_lambda": 0.011658731377413597,
#            "seed": 42,
#        }
#        mlflow.set_tag("model", "xgboost")
#       mlflow.log_params(best_params)

#        booster = xgb.train(
#            params=best_params,
#            dtrain=train,
#            num_boost_round=100,
#            evals=[(valid, "validation")],
#            early_stopping_rounds=50,
#        )

#        y_pred = booster.predict(valid)
        #rmse = mean_squared_error(y_val, y_pred, squared=False)
#        mape = mean_absolute_percentage_error(y_val, y_pred)
        #mlflow.log_metric("rmse", rmse)
#        mlflow.log_metric("mape", mape)

        #pathlib.Path("models").mkdir(exist_ok=True)
        #with open("models/preprocessor.b", "wb") as f_out:
         #   pickle.dump(dv, f_out)
        #mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

#        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
#    return None

@task(log_prints=True)
def train_best_model(X_train,X_val,y_train,y_val, dv):

    mlflow.sklearn.autolog()

    for model_class in (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR):

        with mlflow.start_run():

            #mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
            #mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")
            #mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
            mlflow.set_tag("model", "models")

            mlmodel = model_class()
            mlmodel.fit(X_train, y_train)

            y_pred = mlmodel.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mape", mape)
            pathlib.Path("models").mkdir(exist_ok=True)
            with open("models/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
    return None




@flow
def main_flow(Data_path='./Data/Finall_Data.csv'):
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Forecast")

    # Load
    df = read_data(Data_path)
    df = convert_date(df)

    # Feature Encoding
    df = feature_encoding(df)

    # Clean Data

    df = clean_data(df)


    # Feature Extraction

    df = feature_extraction(df)


    # Fill Null Value
    df = fill_null_value(df)
    
    #Split Data
    X_train,y_train,X_valid,y_valid,X_test,y_test,test, dv = split_data(df)
    save_data(test)

    # Train
    train_best_model(X_train,X_valid,y_train,y_valid,dv)


    # Transform
    #X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Train
    #train_best_model(X_train, X_val, y_train, y_val, dv)

if __name__ == "__main__":
    main_flow()