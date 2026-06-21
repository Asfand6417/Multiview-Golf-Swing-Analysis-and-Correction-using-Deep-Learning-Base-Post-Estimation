import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


class CalibrationError(ValueError):
    """Raised when real checkerboard calibration cannot be completed."""


def _pattern_size(board_rows, board_cols):
    rows = int(board_rows)
    cols = int(board_cols)
    if rows <= 0 or cols <= 0:
        raise CalibrationError("Checkerboard rows and columns must be positive.")
    return cols, rows


def _object_points(board_rows, board_cols, square_size_mm):
    square_size = float(square_size_mm)
    if square_size <= 0:
        raise CalibrationError("Checkerboard square size must be positive.")

    cols, rows = _pattern_size(board_rows, board_cols)
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return points


def _detect_checkerboard(image_path, board_rows, board_cols):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = _pattern_size(board_rows, board_cols)
    found, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    image_size = (image.shape[1], image.shape[0])
    return refined, image_size


def detect_checkerboard_points(image_paths, board_rows, board_cols, square_size_mm):
    """Detect and refine checkerboard corners in a list of calibration images."""
    object_template = _object_points(board_rows, board_cols, square_size_mm)
    object_points = []
    image_points = []
    valid_images = []
    image_size = None

    for image_path in image_paths:
        detected = _detect_checkerboard(image_path, board_rows, board_cols)
        if detected is None:
            continue

        corners, detected_size = detected
        if image_size is None:
            image_size = detected_size
        elif detected_size != image_size:
            raise CalibrationError("Calibration images must have a consistent image size.")

        object_points.append(object_template.copy())
        image_points.append(corners)
        valid_images.append(str(image_path))

    return {
        "object_points": object_points,
        "image_points": image_points,
        "image_size": image_size,
        "valid_images": valid_images,
    }


def calibrate_single_camera(object_points, image_points, image_size):
    if not object_points or not image_points:
        raise CalibrationError("No valid checkerboard detections for single-camera calibration.")
    if image_size is None:
        raise CalibrationError("Image size is unavailable for calibration.")

    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        tuple(image_size),
        None,
        None,
    )
    if not np.isfinite(rms):
        raise CalibrationError("Single-camera reprojection error is not finite.")

    return {
        "K": camera_matrix,
        "D": distortion,
        "reprojection_error": float(rms),
        "image_size": list(image_size),
    }


def build_projection_matrices(K_front, K_side, R, T):
    front_extrinsic = np.hstack((np.eye(3), np.zeros((3, 1))))
    side_extrinsic = np.hstack((np.asarray(R, dtype=np.float64), np.asarray(T, dtype=np.float64).reshape(3, 1)))
    P_front = np.asarray(K_front, dtype=np.float64) @ front_extrinsic
    P_side = np.asarray(K_side, dtype=np.float64) @ side_extrinsic
    return P_front, P_side


def calibrate_stereo_cameras(front_image_paths, side_image_paths, board_rows, board_cols, square_size_mm):
    if len(front_image_paths) != len(side_image_paths):
        raise CalibrationError("Front and side calibration image counts must match.")

    object_template = _object_points(board_rows, board_cols, square_size_mm)
    object_points = []
    front_points = []
    side_points = []
    image_size = None

    for front_path, side_path in zip(front_image_paths, side_image_paths):
        front_detected = _detect_checkerboard(front_path, board_rows, board_cols)
        side_detected = _detect_checkerboard(side_path, board_rows, board_cols)
        if front_detected is None or side_detected is None:
            continue

        front_corners, front_size = front_detected
        side_corners, side_size = side_detected
        if front_size != side_size:
            raise CalibrationError("Paired front/side calibration images must share the same size.")
        if image_size is None:
            image_size = front_size
        elif front_size != image_size:
            raise CalibrationError("Calibration image pairs must have a consistent image size.")

        object_points.append(object_template.copy())
        front_points.append(front_corners)
        side_points.append(side_corners)

    valid_pair_count = len(object_points)
    if valid_pair_count == 0:
        raise CalibrationError("No valid paired checkerboard detections were found.")

    front = calibrate_single_camera(object_points, front_points, image_size)
    side = calibrate_single_camera(object_points, side_points, image_size)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms, K_front, D_front, K_side, D_side, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        front_points,
        side_points,
        front["K"],
        front["D"],
        side["K"],
        side["D"],
        tuple(image_size),
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    if not np.isfinite(rms):
        raise CalibrationError("Stereo reprojection error is not finite.")
    if np.linalg.norm(T) <= 1e-9:
        raise CalibrationError("Stereo baseline is too close to zero.")

    P_front, P_side = build_projection_matrices(K_front, K_side, R, T)
    if P_front.shape != (3, 4) or P_side.shape != (3, 4):
        raise CalibrationError("Projection matrices must be 3x4.")

    return {
        "K_front": K_front,
        "D_front": D_front,
        "K_side": K_side,
        "D_side": D_side,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "P_front": P_front,
        "P_side": P_side,
        "reprojection_error": float(rms),
        "valid_pair_count": valid_pair_count,
        "image_size": list(image_size),
    }


def calibration_to_json(calibration, board_rows, board_cols, square_size_mm, calibration_id=None):
    calibration_id = calibration_id or str(uuid4())
    return {
        "calibration_id": calibration_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "board_rows": int(board_rows),
        "board_cols": int(board_cols),
        "square_size_mm": float(square_size_mm),
        "image_size": calibration["image_size"],
        "K_front": np.asarray(calibration["K_front"]).tolist(),
        "D_front": np.asarray(calibration["D_front"]).tolist(),
        "K_side": np.asarray(calibration["K_side"]).tolist(),
        "D_side": np.asarray(calibration["D_side"]).tolist(),
        "R": np.asarray(calibration["R"]).tolist(),
        "T": np.asarray(calibration["T"]).tolist(),
        "E": np.asarray(calibration["E"]).tolist(),
        "F": np.asarray(calibration["F"]).tolist(),
        "P_front": np.asarray(calibration["P_front"]).tolist(),
        "P_side": np.asarray(calibration["P_side"]).tolist(),
        "reprojection_error": float(calibration["reprojection_error"]),
        "valid_pair_count": int(calibration["valid_pair_count"]),
    }


def save_calibration(calibration_payload, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    calibration_file = output_path / f"{calibration_payload['calibration_id']}.json"
    with calibration_file.open("w", encoding="utf-8") as file_handle:
        json.dump(calibration_payload, file_handle, indent=2)
    return calibration_file
