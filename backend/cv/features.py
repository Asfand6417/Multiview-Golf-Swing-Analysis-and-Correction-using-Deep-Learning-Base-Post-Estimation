import numpy as np


def _joint(frame, name):
    value = frame.get("joints", {}).get(name)
    if value is None:
        return None
    xyz = np.array([value["x"], value["y"], value["z"]], dtype=np.float64)
    if not np.all(np.isfinite(xyz)):
        return None
    return xyz


def _angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-12:
        return None
    cosine = np.clip(float(np.dot(ba, bc) / denom), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rotation_degrees(left, right):
    if left is None or right is None:
        return None
    vector = right - left
    if np.linalg.norm(vector) <= 1e-12:
        return None
    return float(np.degrees(np.arctan2(vector[2], vector[0])))


def _center(*points):
    present = [point for point in points if point is not None]
    if len(present) != len(points):
        return None
    return np.mean(np.array(present, dtype=np.float64), axis=0)


def _spine_angle_degrees(hip_center, shoulder_center):
    if hip_center is None or shoulder_center is None:
        return None
    vector = shoulder_center - hip_center
    length = np.linalg.norm(vector)
    if length <= 1e-12:
        return None
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cosine = np.clip(float(np.dot(vector, vertical) / length), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _derivative(current_value, previous_value, current_timestamp, previous_timestamp):
    if current_value is None or previous_value is None:
        return None
    if current_timestamp is None or previous_timestamp is None:
        return None
    delta_t = float(current_timestamp) - float(previous_timestamp)
    if delta_t <= 0:
        return None
    return float((current_value - previous_value) / delta_t)


def _velocity(current_point, previous_point, current_timestamp, previous_timestamp):
    if current_point is None or previous_point is None:
        return None
    if current_timestamp is None or previous_timestamp is None:
        return None
    delta_t = float(current_timestamp) - float(previous_timestamp)
    if delta_t <= 0:
        return None
    return float(np.linalg.norm(current_point - previous_point) / delta_t)


def _finite_or_none(value):
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None


def extract_biomechanical_features(frames3d):
    """
    Extract per-frame 3D swing features from observed joints only.

    Dependent features are null whenever any required joint is missing.
    """
    feature_frames = []
    first_head = None
    first_pelvis = None
    previous = {
        "timestamp": None,
        "hand_center": None,
        "shoulder_rotation_deg": None,
        "hip_rotation_deg": None,
        "x_factor_deg": None,
    }

    for frame in frames3d:
        timestamp = float(frame.get("timestamp") or 0.0)
        left_shoulder = _joint(frame, "left_shoulder")
        right_shoulder = _joint(frame, "right_shoulder")
        left_elbow = _joint(frame, "left_elbow")
        right_elbow = _joint(frame, "right_elbow")
        left_wrist = _joint(frame, "left_wrist")
        right_wrist = _joint(frame, "right_wrist")
        left_hip = _joint(frame, "left_hip")
        right_hip = _joint(frame, "right_hip")
        left_knee = _joint(frame, "left_knee")
        right_knee = _joint(frame, "right_knee")
        left_ankle = _joint(frame, "left_ankle")
        right_ankle = _joint(frame, "right_ankle")
        head = _joint(frame, "nose")

        shoulder_center = _center(left_shoulder, right_shoulder)
        pelvis_center = _center(left_hip, right_hip)
        hand_center = _center(left_wrist, right_wrist)

        if first_head is None and head is not None:
            first_head = head
        if first_pelvis is None and pelvis_center is not None:
            first_pelvis = pelvis_center

        shoulder_rotation = _rotation_degrees(left_shoulder, right_shoulder)
        hip_rotation = _rotation_degrees(left_hip, right_hip)
        x_factor = (
            float(shoulder_rotation - hip_rotation)
            if shoulder_rotation is not None and hip_rotation is not None
            else None
        )
        wrist_height = (
            float(hand_center[1] - shoulder_center[1])
            if hand_center is not None and shoulder_center is not None
            else None
        )

        features = {
            "shoulder_rotation_deg": _finite_or_none(shoulder_rotation),
            "hip_rotation_deg": _finite_or_none(hip_rotation),
            "x_factor_deg": _finite_or_none(x_factor),
            "left_elbow_angle_deg": _finite_or_none(_angle(left_shoulder, left_elbow, left_wrist)),
            "right_elbow_angle_deg": _finite_or_none(_angle(right_shoulder, right_elbow, right_wrist)),
            "left_knee_angle_deg": _finite_or_none(_angle(left_hip, left_knee, left_ankle)),
            "right_knee_angle_deg": _finite_or_none(_angle(right_hip, right_knee, right_ankle)),
            "spine_angle_deg": _finite_or_none(_spine_angle_degrees(pelvis_center, shoulder_center)),
            "wrist_height": _finite_or_none(wrist_height),
            "hand_velocity": _finite_or_none(_velocity(
                hand_center,
                previous["hand_center"],
                timestamp,
                previous["timestamp"],
            )),
            "pelvis_displacement": (
                _finite_or_none(np.linalg.norm(pelvis_center - first_pelvis))
                if pelvis_center is not None and first_pelvis is not None
                else None
            ),
            "head_sway": (
                _finite_or_none(head[0] - first_head[0])
                if head is not None and first_head is not None
                else None
            ),
            "shoulder_angular_velocity_deg_s": _finite_or_none(_derivative(
                shoulder_rotation,
                previous["shoulder_rotation_deg"],
                timestamp,
                previous["timestamp"],
            )),
            "hip_angular_velocity_deg_s": _finite_or_none(_derivative(
                hip_rotation,
                previous["hip_rotation_deg"],
                timestamp,
                previous["timestamp"],
            )),
            "x_factor_velocity_deg_s": _finite_or_none(_derivative(
                x_factor,
                previous["x_factor_deg"],
                timestamp,
                previous["timestamp"],
            )),
        }

        feature_frames.append({
            "frame_index": frame.get("frame_index"),
            "timestamp": timestamp,
            "features": features,
        })

        previous = {
            "timestamp": timestamp,
            "hand_center": hand_center,
            "shoulder_rotation_deg": shoulder_rotation,
            "hip_rotation_deg": hip_rotation,
            "x_factor_deg": x_factor,
        }

    return feature_frames


def any_feature_available(feature_frames):
    for frame in feature_frames:
        if any(value is not None for value in frame.get("features", {}).values()):
            return True
    return False
