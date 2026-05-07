import np

from multicamera.DL_Model.landmark_feature_preparation import prepare_features_for_model


def predict_mpjpe(lmList, mpjpe_model=None):
    # Prepare features for the model
    features = prepare_features_for_model(lmList)
    
    # Make prediction using the trained model
    mpjpe_pred = float(mpjpe_model.predict(np.array([features]))[0][0])
    return mpjpe_pred