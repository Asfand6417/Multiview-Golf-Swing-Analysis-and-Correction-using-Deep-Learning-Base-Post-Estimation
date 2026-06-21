from pathlib import Path

import numpy as np

try:
    import joblib
except ModuleNotFoundError:
    joblib = None

try:
    from cv.features import extract_biomechanical_features
except ModuleNotFoundError:
    from backend.cv.features import extract_biomechanical_features


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "swing_phase_svm.joblib"
FEATURE_NAMES = [
    "wrist_height",
    "hand_velocity",
    "shoulder_rotation_deg",
    "hip_rotation_deg",
    "x_factor_deg",
    "left_knee_angle_deg",
]
EIGHT_PHASES = [
    "Address",
    "Takeaway",
    "Backswing",
    "Top",
    "Transition",
    "Downswing",
    "Impact",
    "Follow-through",
]


def _joint(frame, name):
    value = frame.get("joints", {}).get(name)
    if value is None:
        return None
    return np.array([value["x"], value["y"], value["z"]], dtype=np.float64)


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-12:
        return None
    cosine = np.clip(float(np.dot(ba, bc) / denom), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rotation_degrees(left, right):
    vector = right - left
    return float(np.degrees(np.arctan2(vector[2], vector[0])))


def extract_phase_features(frames3d):
    if frames3d and "features" in frames3d[0]:
        return _extract_svm_features_from_feature_frames(frames3d)

    feature_frames = extract_biomechanical_features(frames3d)
    return _extract_svm_features_from_feature_frames(feature_frames)


def _extract_svm_features_from_feature_frames(feature_frames):
    features = []

    for frame in feature_frames:
        source_values = frame.get("features", {})
        values = {name: source_values.get(name) for name in FEATURE_NAMES}

        if all(values[name] is not None and np.isfinite(values[name]) for name in FEATURE_NAMES):
            features.append({
                "frame_index": frame.get("frame_index"),
                "timestamp": float(frame.get("timestamp") or 0.0),
                "values": values,
            })

    return features


def segment_phases_with_svm(frames3d, model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        return {
            "available": False,
            "method": None,
            "phases": [],
            "notes": "No trained SVM model found.",
        }
    if joblib is None:
        return {
            "available": False,
            "method": None,
            "phases": [],
            "notes": "joblib is required to load a trained SVM model.",
        }

    features = extract_phase_features(frames3d)
    if not features:
        return {
            "available": False,
            "method": None,
            "phases": [],
            "notes": "Not enough observed 3D joints to compute phase features.",
        }

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        model = loaded.get("model")
        scaler = loaded.get("scaler")
    else:
        model = loaded
        scaler = None

    if model is None:
        return {
            "available": False,
            "method": None,
            "phases": [],
            "notes": "SVM model file did not contain a usable model.",
        }

    matrix = np.array([[entry["values"][name] for name in FEATURE_NAMES] for entry in features], dtype=np.float64)
    if scaler is not None:
        matrix = scaler.transform(matrix)

    predictions = model.predict(matrix)

    def normalize_phase(prediction):
        if isinstance(prediction, (int, np.integer)) and 0 <= int(prediction) < len(EIGHT_PHASES):
            return EIGHT_PHASES[int(prediction)]
        return str(prediction)

    phases = [
        {
            "frame_index": entry["frame_index"],
            "timestamp": entry["timestamp"],
            "phase": normalize_phase(prediction),
        }
        for entry, prediction in zip(features, predictions)
    ]
    return {
        "available": True,
        "method": "svm_rbf",
        "phases": phases,
        "notes": "Predicted with trained SVM model. Numeric labels 0-7 are mapped to the eight golf swing phases.",
    }
