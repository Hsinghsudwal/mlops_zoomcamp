from prefect import flow, task
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import tempfile
import pickle
import os

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
def split_data(df: pd.DataFrame):
    df_train = df.copy()
    df_val = df.copy()
    return df_train, df_val

@task
def train_and_log_model(df_train, df_val):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    target = 'duration'

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

    with open("model.pkl", 'wb') as f_in:
        pickle.dump(model,f_in)
    
    # Log with MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("yellow-taxi-regression")

    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path_model = os.path.join(tmp_dir, "model.pkl")
            path_dv = os.path.join(tmp_dir, "dv.pkl")
            
            with open(path_model, "wb") as f_out:
                pickle.dump(model, f_out)
            with open(path_dv, "wb") as f_out:
                pickle.dump(dv, f_out)

            mlflow.log_artifact(path_model, artifact_path="model")
            mlflow.log_artifact(path_dv, artifact_path="preprocessor")


@flow
def main():
    df = read_data(DATA_URL)
    df_train, df_val = split_data(df)
    train_and_log_model(df_train, df_val)

if __name__ == "__main__":
    main()
