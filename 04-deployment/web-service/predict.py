import pickle

with open('../../week02_Experiment_tracking_and_model_management/models/lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]
