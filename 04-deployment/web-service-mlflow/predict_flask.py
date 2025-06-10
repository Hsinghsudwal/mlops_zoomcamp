import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify

MLFLOW_TRACKING_URI = 'sqlite:///mlruns.db'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RUN_ID = 'e60b4352b7134f679cb04d95288a24c0'
logged_model = f'runs:/{RUN_ID}/model'

model = mlflow.pyfunc.load_model(logged_model)

#client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

#path = client.download_artifacts(run_id=RUN_ID, path='model/model.pkl')
#print(f'downloading the pipeline to {path}')
#with open(path, 'rb') as f_out:
#    pipeline = pickle.load(f_out)

#dv = pipeline[0]
#model = pipeline[1]

def prepare_features(ride):
    features = {}
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    #X = dv.transform(features)
    preds = model.predict(features)
    return preds[0]


# flask app
app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # get data
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
            'duration': pred,
            'model_version': RUN_ID
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
