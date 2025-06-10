#!/usr/bin/env python
# coding: utf-8

import os
import sys

import pickle
import uuid
from datetime import datetime

import pandas as pd
import mlflow
from prefect import task, flow, get_run_logger
from prefect.context import get_run_context
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline



MLFLOW_TRACKING_URI = 'sqlite:///mlruns.db'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# create unique ids
def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids
    
def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() /60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = generate_uuids(len(df))

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df


def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def load_model(run_id):
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def save_results(df, y_pred, run_id, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    
    df_result.to_parquet(output_file, index=False)
    return output_file



@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()
    logger.info(f'reading the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    
    logger.info(f'loading the model with RUN_ID={run_id}...')
    model = load_model(run_id)

    logger.info(f'applying the model...')
    y_pred = model.predict(dicts)
    
    logger.info(f'saving the results to {output_file}...')
    save_results(df, y_pred, run_id, output_file)
    return output_file

def get_paths(run_date, run_id):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month
    #input_file = '../../data/green_tripdata_2021-01.parquet'
    input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_{year:04}-{month:02d}.parquet'
    #input_file = f's3://nyc-tlc/trip data/green_tripdata_{year:04}-{month:02d}.parquet'
    #output_file = f's3://<name-of-s3-bucket>/green_tripdata/year={year:04}/month={month:02d}/{run_id}.parquet'
    output_file = f'../../output/green_tripdata_{year:04}-{month:02d}.parquet'
    
    return input_file, output_file


@flow
def ride_duration_prediction(run_id: str, run_date: datetime=None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow.expected_start_time

    input_file, output_file = get_paths(run_date, run_id)
   
    apply_model(input_file=input_file, run_id=run_id, output_file=output_file)

def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3

    #RUN_ID = '54c0663c677e417f8900b51ed7985878'
    run_id = sys.argv[3] 

    ride_duration_prediction(run_id=run_id, run_date=datetime(year=year, month=month, day=1))    

if __name__ == '__main__':
    run()
