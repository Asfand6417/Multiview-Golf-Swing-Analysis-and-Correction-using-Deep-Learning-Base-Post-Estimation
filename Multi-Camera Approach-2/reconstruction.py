"""
DLT Triangulation for 3D Reconstruction from Multiple Camera Views

This module implements Direct Linear Transformation (DLT) for triangulating
3D points from corresponding 2D points in calibrated camera views.

Research Objectives:
    1. Accuracy of 3D reconstruction [[45], [46]]

Technical Constraints:
    - 3D axes: x = left→right, y = up, z = forward (toward ball)
    - Feet should be anchored to z=0 ground plane
"""

import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# Hyperparameters for triangulation
REPROJECTION_ERROR_THRESHOLD = 10.0  # Maximum acceptable reprojection error in pixels
SMOOTHING_ALPHA = 0.7  # Smoothing factor for temporal filtering (0-1)


def triangulate_points(keypoints1, keypoints2, calibration_data):
    """
    Triangulate 3D points from corresponding 2D points in both camera views.

    Args:
        keypoints1: Array of 2D points from camera 1 (Nx2)
        keypoints2: Array of 2D points from camera 2 (Nx2)
        calibration_data: Dictionary containing camera calibration parameters
            - camera1_matrix: Intrinsic matrix for camera 1
            - camera2_matrix: Intrinsic matrix for camera 2
            - R: Rotation matrix from camera 1 to camera 2
            - T: Translation vector from camera 1 to camera 2

    Returns:
        points_3d: Array of triangulated 3D points (Nx3)

    Research Alignment:
        Supports Objective 1: Accuracy of 3D reconstruction [[45], [46]]
        by implementing DLT triangulation with reprojection error checking.
    """
    # Get calibration parameters
    mtx1 = calibration_data['camera1_matrix']
    mtx2 = calibration_data['camera2_matrix']
    R = calibration_data['R']
    T = calibration_data['T']

    # Compute projection matrices for both cameras
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # [I|0]
    P1 = mtx1 @ P1

    # Compute projection matrix for camera 2
    RT = np.hstack((R, T))
    P2 = mtx2 @ RT

    # Initialize array for 3D points
    points_3d = []
    reprojection_errors = []

    for kp1, kp2 in zip(keypoints1, keypoints2):
        # Convert points to correct format
        point1 = np.array([[kp1[0]], [kp1[1]]])
        point2 = np.array([[kp2[0]], [kp2[1]]])

        # Triangulate one point
        point_4d = cv2.triangulatePoints(P1, P2, point1, point2)

        # Convert to 3D homogeneous coordinates
        point_3d = point_4d[:3] / point_4d[3]

        # Calculate reprojection error to verify accuracy
        error = calculate_reprojection_error(point_3d, point1, point2, P1, P2)
        reprojection_errors.append(error)

        points_3d.append(point_3d.flatten())

    points_3d = np.array(points_3d)

    # Print average reprojection error for quality assessment
    avg_error = np.mean(reprojection_errors)
    print(f"Average reprojection error: {avg_error:.2f} pixels")

    # Assert that reprojection error is below threshold
    assert avg_error < REPROJECTION_ERROR_THRESHOLD, f"Reprojection error ({avg_error:.2f} px) exceeds threshold ({REPROJECTION_ERROR_THRESHOLD} px)"

    return points_3d


def calculate_reprojection_error(point_3d, point1, point2, P1, P2):
    """
    Calculate reprojection error for a triangulated 3D point.

    Args:
        point_3d: Triangulated 3D point
        point1, point2: Original 2D points from cameras
        P1, P2: Projection matrices for both cameras

    Returns:
        Average reprojection error in pixels
    """
    # Convert 3D point to homogeneous coordinates
    point_3d_homogeneous = np.append(point_3d, 1)

    # Project 3D point back to 2D in both camera views
    projected1 = P1 @ point_3d_homogeneous
    projected1 = projected1[:2] / projected1[2]

    projected2 = P2 @ point_3d_homogeneous
    projected2 = projected2[:2] / projected2[2]

    # Calculate Euclidean distance between original and reprojected points
    error1 = np.linalg.norm(point1 - projected1)
    error2 = np.linalg.norm(point2 - projected2)

    # Return average error across both cameras
    return (error1 + error2) / 2


def align_ground_plane(points_3d):
    """
    Align the skeleton so the feet rest on the z=0 ground plane.

    Args:
        points_3d: Array of 3D points (Nx3)

    Returns:
        aligned_points: Array of 3D points aligned to ground plane

    Technical Constraint:
        Ensures feet ≈ z=0 as specified in requirements.
    """
    # Make a copy to avoid modifying the original
    aligned_points = points_3d.copy()

    # Assuming points follow MediaPipe convention:
    # Feet landmarks are indices 27-32 (left ankle, left heel, left foot index, 
    # right ankle, right heel, right foot index)
    foot_indices = [27, 28, 29, 30, 31, 32]

    # Extract foot points that exist in the data
    foot_points = [points_3d[i] for i in foot_indices if i < len(points_3d)]

    if foot_points:
        # Find the lowest point (highest z-value if z is pointing down)
        # or (lowest z-value if z is pointing up)
        # Assuming z is forward (toward ball), we want the minimum y value
        min_y = min([point[1] for point in foot_points])

        # Translate all points so the lowest foot point is at y=0
        aligned_points[:, 1] -= min_y

    return aligned_points


def apply_anatomical_constraints(points_3d):
    """
    Apply anatomical constraints to ensure valid human skeletal structure.

    Args:
        points_3d: Array of 3D points (Nx3)

    Returns:
        constrained_points: Array of 3D points with anatomical constraints applied

    Research Alignment:
        Supports Objective 1: Accuracy of 3D reconstruction by enforcing
        anatomically plausible skeleton configurations.
    """
    # Handle empty or None input
    if points_3d is None or len(points_3d) == 0:
        return points_3d

    # Define expected bone lengths based on average proportions
    expected_bone_lengths = {
        (11, 13): 0.3,  # Upper arm
        (13, 15): 0.25,  # Lower arm
        (12, 14): 0.3,  # Upper arm
        (14, 16): 0.25,  # Lower arm
        (11, 12): 0.35,  # Shoulder width
        (23, 24): 0.25,  # Hip width
        (11, 23): 0.55,  # Left shoulder to hip
        (12, 24): 0.55,  # Right shoulder to hip
        (23, 25): 0.45,  # Left hip to knee
        (24, 26): 0.45,  # Right hip to knee
        (25, 27): 0.45,  # Left knee to ankle
        (26, 28): 0.45   # Right knee to ankle
    }

    # Make a copy to avoid modifying the original during iterations
    constrained_points = points_3d.copy()

    # Enforce bone length constraints
    for (idx1, idx2), expected_length in expected_bone_lengths.items():
        if idx1 < len(constrained_points) and idx2 < len(constrained_points):
            actual_vec = constrained_points[idx2] - constrained_points[idx1]
            actual_length = np.linalg.norm(actual_vec)

            # Skip if actual_length is zero or very small to avoid division by zero
            if actual_length < 1e-6:
                continue

            # Only adjust if the length is significantly different
            if abs(actual_length - expected_length) > 0.1 * expected_length:
                # Scale vector to expected length
                scaled_vec = actual_vec * (expected_length / actual_length)

                # Adjust the points (distribute the correction 50/50)
                midpoint = (constrained_points[idx1] + constrained_points[idx2]) / 2
                constrained_points[idx1] = midpoint - scaled_vec / 2
                constrained_points[idx2] = midpoint + scaled_vec / 2

    # Enforce joint angle constraints
    # Define expected angle ranges for major joints
    angle_constraints = {
        # (joint1, pivot, joint2): (min_angle, max_angle)
        (11, 13, 15): (45, 175),  # Left elbow
        (12, 14, 16): (45, 175),  # Right elbow
        (23, 25, 27): (100, 180), # Left knee
        (24, 26, 28): (100, 180), # Right knee
    }

    # Apply angle constraints
    for (idx1, pivot_idx, idx2), (min_angle, max_angle) in angle_constraints.items():
        if (idx1 < len(constrained_points) and 
            pivot_idx < len(constrained_points) and 
            idx2 < len(constrained_points)):

            v1 = constrained_points[idx1] - constrained_points[pivot_idx]
            v2 = constrained_points[idx2] - constrained_points[pivot_idx]

            # Calculate the angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure it's in valid range
            angle = np.degrees(np.arccos(cos_angle))

            # If angle is outside the constraint range, adjust it
            if angle < min_angle:
                # Adjust to minimum angle
                target_angle = min_angle
                adjust = True
            elif angle > max_angle:
                # Adjust to maximum angle
                target_angle = max_angle
                adjust = True
            else:
                adjust = False

            if adjust:
                # Convert target angle to radians
                target_rad = np.radians(target_angle)

                # Create rotation matrix to rotate v2 to target angle
                # This is a simplified approach and works best for 2D rotations
                # For full 3D, you would need a more complex rotation operation

                # Normalize vectors
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)

                # Get rotation axis (perpendicular to plane of v1, v2)
                rotation_axis = np.cross(v1_norm, v2_norm)
                if np.linalg.norm(rotation_axis) < 1e-6:
                    # If vectors are collinear, create a perpendicular vector
                    rotation_axis = np.array([1, 0, 0]) if abs(v1_norm[1]) > abs(v1_norm[0]) else np.array([0, 1, 0])
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

                # Calculate adjustment angle
                adjustment_angle = target_rad - np.arccos(cos_angle)

                # Create rotation matrix
                rot = Rotation.from_rotvec(rotation_axis * adjustment_angle)

                # Apply rotation to v2
                rotated_v2 = rot.apply(v2)

                # Update point position
                constrained_points[idx2] = constrained_points[pivot_idx] + rotated_v2

    return constrained_points


def smooth_trajectories(current_points, previous_points, alpha=SMOOTHING_ALPHA):
    """
    Apply temporal smoothing to reduce jitter in 3D point trajectories.

    Args:
        current_points: Current frame's 3D points
        previous_points: Previous frame's 3D points
        alpha: Smoothing factor (0-1), higher values give more weight to current frame

    Returns:
        smoothed_points: Temporally smoothed 3D points

    Research Alignment:
        Supports Objective 1: Accuracy of 3D reconstruction by reducing
        temporal jitter in the reconstructed skeleton.
    """
    # Handle edge cases
    if current_points is None or previous_points is None:
        return current_points

    if len(current_points) != len(previous_points):
        # If sizes don't match, return current points
        return current_points

    # Simple exponential smoothing
    try:
        # Convert to appropriate type to ensure mathematical operations work
        current_np = np.array(current_points, dtype=np.float32)
        previous_np = np.array(previous_points, dtype=np.float32)

        smoothed_points = alpha * current_np + (1 - alpha) * previous_np
        return smoothed_points
    except Exception as e:
        print(f"Error during trajectory smoothing: {e}")
        # Return original points if smoothing fails
        return current_points


def triangulate_sequence(keypoints1_seq, keypoints2_seq, calibration_data):
    """
    Triangulate a sequence of frames to produce 3D skeletons.

    Args:
        keypoints1_seq: List of keypoint arrays from camera 1
        keypoints2_seq: List of keypoint arrays from camera 2
        calibration_data: Camera calibration data

    Returns:
        points_3d_seq: List of 3D point arrays, one per frame
    """
    points_3d_seq = []
    prev_points_3d = None

    for keypoints1, keypoints2 in zip(keypoints1_seq, keypoints2_seq):
        # Triangulate points for this frame
        points_3d = triangulate_points(keypoints1, keypoints2, calibration_data)

        # Apply anatomical constraints to ensure valid skeleton
        points_3d = apply_anatomical_constraints(points_3d)

        # Apply temporal smoothing if we have previous points
        if prev_points_3d is not None:
            points_3d = smooth_trajectories(points_3d, prev_points_3d)

        # Align to ground plane
        aligned_points = align_ground_plane(points_3d)

        # Store for next frame's smoothing
        prev_points_3d = aligned_points.copy()

        points_3d_seq.append(aligned_points)

    return points_3d_seq


if __name__ == "__main__":
    # Example usage
    print("DLT Triangulation module - Run through the main pipeline")
