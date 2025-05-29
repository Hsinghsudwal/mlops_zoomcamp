from prefect import flow, task
import pandas as pd

# Yellow Taxi March 2023 data
DATA_URL = "data/yellow_tripdata_2023-03.parquet"

@task
def read_taxi_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Raw data loaded: {len(df)}")
    return df

@task
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Data after preparation: {len(df)}")
    return df

@flow
def yellow_taxi_pipeline():
    df_raw = read_taxi_data(DATA_URL)
    df_prepared = prepare_data(df_raw)
    return df_prepared

if __name__ == "__main__":
    yellow_taxi_pipeline()
