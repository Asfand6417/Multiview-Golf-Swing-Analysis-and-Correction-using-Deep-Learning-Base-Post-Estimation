import cv2
import numpy as np

def reconstruct_3d(pts_front, pts_side, P_front, P_side):
    """
    Convert synchronized 2D points from two views into 3D points.
    :param pts_front: 2xN numpy array of MediaPipe pose points (front camera)
    :param pts_side: 2xN numpy array of MediaPipe pose points (side camera)
    :param P_front: 3x4 Camera Projection Matrix for front camera
    :param P_side: 3x4 Camera Projection Matrix for side camera
    :return: Nx3 numpy array of reconstructed 3D points
    """
    # Use Direct Linear Transformation (DLT) for triangulation
    pts_4d_homogeneous = cv2.triangulatePoints(P_front, P_side, pts_front, pts_side)
    
    # Normalize homogeneous coordinates back to 3D Cartesian coordinates
    pts_3d = pts_4d_homogeneous[:3, :] / pts_4d_homogeneous[3, :]
    
    return pts_3d.T


def _valid_2d_point(point):
    if point is None or len(point) != 2:
        return False
    try:
        values = np.asarray(point, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def triangulate_joint(point_front, point_side, P_front, P_side):
    """
    Triangulate one joint from two calibrated 2D observations.

    Returns None when either view is missing or the homogeneous result is
    invalid. Missing demo observations must remain missing in the API output.
    """
    if not (_valid_2d_point(point_front) and _valid_2d_point(point_side)):
        return None

    front = np.asarray(point_front, dtype=np.float64).reshape(2, 1)
    side = np.asarray(point_side, dtype=np.float64).reshape(2, 1)
    front_projection = np.asarray(P_front, dtype=np.float64)
    side_projection = np.asarray(P_side, dtype=np.float64)

    point_4d = cv2.triangulatePoints(front_projection, side_projection, front, side).reshape(4)
    w = point_4d[3]
    if not np.isfinite(w) or abs(w) < 1e-12:
        return None

    xyz = point_4d[:3] / w
    if not np.all(np.isfinite(xyz)):
        return None

    return {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
        "observed": True
    }


def triangulate_dlt_point(point_front, point_side, P_front, P_side):
    return triangulate_joint(point_front, point_side, P_front, P_side)


def triangulate_frame(frame_pair, joint_names, P_front, P_side):
    front_points = frame_pair.get("front", {}).get("points", {})
    side_points = frame_pair.get("side", {}).get("points", {})
    joints = {}

    for joint_name in joint_names:
        joints[joint_name] = triangulate_joint(
            front_points.get(joint_name),
            side_points.get(joint_name),
            P_front,
            P_side,
        )

    return {
        "frame_index": frame_pair.get("frame_index"),
        "timestamp": frame_pair.get("timestamp"),
        "joints": joints
    }


def triangulate_sequence(paired_frames, joint_names, P_front, P_side):
    return [
        triangulate_frame(frame_pair, joint_names, P_front, P_side)
        for frame_pair in paired_frames
    ]


def reconstruct_dlt_frames(demo_frames, joint_names, P_front, P_side):
    """
    Reconstruct frame-by-frame joints from stored paired 2D demo observations.

    The returned shape is designed for the frontend and preserves missing joints
    as None instead of interpolating or hallucinating positions.
    """
    reconstructed_frames = []

    for frame in demo_frames:
        front_points = frame.get("points_front", [])
        side_points = frame.get("points_side", [])
        joints = {}

        for joint_index, joint_name in enumerate(joint_names):
            point_front = front_points[joint_index] if joint_index < len(front_points) else None
            point_side = side_points[joint_index] if joint_index < len(side_points) else None
            joints[joint_name] = triangulate_joint(point_front, point_side, P_front, P_side)

        reconstructed_frames.append({
            "frame_index": frame.get("frame_index"),
            "timestamp": frame.get("timestamp"),
            "joints": joints
        })

    return reconstructed_frames
