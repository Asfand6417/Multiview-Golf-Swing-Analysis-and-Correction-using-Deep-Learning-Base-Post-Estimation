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
