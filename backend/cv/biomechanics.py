try:
    from cv.features import any_feature_available, extract_biomechanical_features
except ModuleNotFoundError:
    from backend.cv.features import any_feature_available, extract_biomechanical_features


def _feature_frames(input_data):
    if input_data and "features" in input_data[0]:
        return input_data
    return extract_biomechanical_features(input_data)


def _phase_by_frame(phase_segmentation):
    if not phase_segmentation or not phase_segmentation.get("available"):
        return {}
    return {
        phase["frame_index"]: phase.get("phase")
        for phase in phase_segmentation.get("phases", [])
    }


def _append_error(errors, frame, phase, error, severity, evidence, suggestion):
    errors.append({
        "frame_index": frame.get("frame_index"),
        "phase": phase,
        "error": error,
        "severity": severity,
        "evidence": evidence,
        "suggestion": suggestion,
    })


def analyze_biomechanics(input_data, phase_segmentation=None):
    """Run rule-based checks from real 3D features and real phase labels."""
    feature_frames = _feature_frames(input_data)
    if not any_feature_available(feature_frames):
        return {
            "available": False,
            "features": feature_frames,
            "errors": [],
            "faults": [],
            "corrections": [],
            "notes": "Not enough observed 3D joints to compute biomechanical feedback.",
        }

    phases = _phase_by_frame(phase_segmentation)
    if not phases:
        return {
            "available": False,
            "features": feature_frames,
            "errors": [],
            "faults": [],
            "corrections": [],
            "notes": "Biomechanical rules require computed 3D features and phase labels.",
        }

    baseline_spine_angle = next(
        (
            frame["features"]["spine_angle_deg"]
            for frame in feature_frames
            if frame.get("features", {}).get("spine_angle_deg") is not None
        ),
        None,
    )
    errors = []

    for frame in feature_frames:
        phase = phases.get(frame.get("frame_index"))
        if not phase:
            continue

        values = frame.get("features", {})
        x_factor = values.get("x_factor_deg")
        spine_angle = values.get("spine_angle_deg")
        left_elbow = values.get("left_elbow_angle_deg")
        left_knee = values.get("left_knee_angle_deg")
        hip_rotation = values.get("hip_rotation_deg")
        pelvis_displacement = values.get("pelvis_displacement")
        head_sway = values.get("head_sway")

        if phase == "Top" and x_factor is not None and abs(x_factor) < 35.0:
            _append_error(
                errors,
                frame,
                phase,
                "X-factor deficiency",
                "moderate",
                {"x_factor_deg": x_factor, "threshold_deg": 35.0},
                "Increase shoulder turn relative to hip turn by the top of backswing.",
            )

        if phase == "Top" and left_elbow is not None and left_elbow < 150.0:
            _append_error(
                errors,
                frame,
                phase,
                "Lead arm extension loss",
                "moderate",
                {"left_elbow_angle_deg": left_elbow, "threshold_deg": 150.0},
                "Maintain a longer lead arm structure near the top without locking the elbow.",
            )

        if phase == "Top" and hip_rotation is not None and abs(hip_rotation) < 20.0:
            _append_error(
                errors,
                frame,
                phase,
                "Hip rotation restriction",
                "moderate",
                {"hip_rotation_deg": hip_rotation, "threshold_deg": 20.0},
                "Allow the trail hip to turn enough to support the backswing coil.",
            )

        if phase == "Impact" and baseline_spine_angle is not None and spine_angle is not None:
            change = abs(spine_angle - baseline_spine_angle)
            if change > 12.0:
                _append_error(
                    errors,
                    frame,
                    phase,
                    "Spine angle loss",
                    "moderate",
                    {"spine_angle_change_deg": change, "threshold_deg": 12.0},
                    "Maintain posture depth through impact instead of standing up early.",
                )

        if phase in {"Downswing", "Impact"} and pelvis_displacement is not None and pelvis_displacement > 120.0:
            _append_error(
                errors,
                frame,
                phase,
                "Early extension",
                "moderate",
                {"pelvis_displacement": pelvis_displacement, "threshold": 120.0},
                "Maintain hip depth through downswing.",
            )

        if phase in {"Downswing", "Impact"} and head_sway is not None and abs(head_sway) > 120.0:
            _append_error(
                errors,
                frame,
                phase,
                "Excessive head sway",
                "moderate",
                {"head_sway": head_sway, "threshold": 120.0},
                "Reduce lateral head travel by rotating around a more stable spine axis.",
            )

        if phase == "Impact" and left_knee is not None and left_knee > 175.0:
            _append_error(
                errors,
                frame,
                phase,
                "Lead knee lockout",
                "moderate",
                {"left_knee_angle_deg": left_knee, "threshold_deg": 175.0},
                "Keep some lead-knee flex into transition before extending through impact.",
            )

    corrections = []
    for error in errors:
        if error["suggestion"] not in corrections:
            corrections.append(error["suggestion"])

    faults = [
        {
            "id": error["error"].lower().replace(" ", "_").replace("-", "_"),
            "label": error["error"],
            "measurement": next(iter(error["evidence"].values())),
            "threshold": list(error["evidence"].values())[-1],
            "unit": "computed_feature",
        }
        for error in errors
    ]

    return {
        "available": True,
        "features": feature_frames,
        "errors": errors,
        "faults": faults,
        "corrections": corrections,
        "notes": (
            "Rule-based feedback computed from observed 3D features and phase labels."
            if errors
            else "No rule violations were found in the available phase-labeled 3D features."
        ),
    }
