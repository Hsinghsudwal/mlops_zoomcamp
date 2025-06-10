#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys


def load_models():

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return dv, model

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def predict_model(df,dv,model):
    # dv,model=load_models()
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

def main():
    
    year = int(sys.argv[1]) #2023
    month = int(sys.argv[2])#04

    filename=f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    print("Reading data")
    df=read_data(filename)

    print("Loading model")
    dv, model = load_models()
    print("Prediction")
    y_pred=predict_model(df, dv, model)
    print(f'prediction mean: {y_pred.mean()}')


if __name__=="__main__":
    main()





