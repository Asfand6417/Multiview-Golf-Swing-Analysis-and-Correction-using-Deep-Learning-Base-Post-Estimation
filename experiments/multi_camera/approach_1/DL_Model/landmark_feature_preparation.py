def prepare_features_for_model(lmList):
    # Convert landmark format from MediaPipe to model input
    features = []
    for lm in lmList:
        # Extract x, y coordinates and normalize
        features.extend([lm[1], lm[2]])
    return np.array(features)