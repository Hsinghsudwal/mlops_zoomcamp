from prefect import flow, task
import pandas as pd

# Local or remote file path
DATA_URL = "data/yellow_tripdata_2023-03.parquet"
@task
def read_taxi_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Data loaded: {len(df)}")
    return df

@flow
def yellow_taxi_pipeline():
    df = read_taxi_data(DATA_URL)
    return df

if __name__ == "__main__":
    yellow_taxi_pipeline()
