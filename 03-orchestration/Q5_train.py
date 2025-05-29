from prefect import flow, task
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data URL for March 2023 Yellow Taxi
DATA_URL = "data/yellow_tripdata_2023-03.parquet"

@task
def read_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    print(f"Data loaded: {len(df)} rows")
    return df

@task
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df

@task
def split_data(df: pd.DataFrame):
    df_train = df.sample(frac=0.8, random_state=42)
    df_val = df.drop(df_train.index)
    return df_train, df_val

@task
def train_model(df_train: pd.DataFrame, df_val: pd.DataFrame):
    target = 'duration'
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df_train[target].values

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_val = df_val[target].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred)

    print(f"Model trained. Intercept: {model.intercept_:.2f}")

    return model, dv

@flow
def yellow_taxi_pipeline():
    df_raw = read_data(DATA_URL)
    df_clean = prepare_data(df_raw)
    df_train, df_val = split_data(df_clean)
    dv, model = train_model(df_train, df_val)

if __name__ == "__main__":
    yellow_taxi_pipeline()