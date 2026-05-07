def triangulate_points(keypoints1, keypoints2, calibration_data):
    """Triangulate 3D points from corresponding 2D points in both camera views"""
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

    for kp1, kp2 in zip(keypoints1, keypoints2):
        # Convert points to correct format
        point1 = np.array([[kp1[0]], [kp1[1]]])
        point2 = np.array([[kp2[0]], [kp2[1]]])

        # Triangulate one point
        point_4d = cv2.triangulatePoints(P1, P2, point1, point2)

        # Convert to 3D homogeneous coordinates
        point_3d = point_4d[:3] / point_4d[3]

        points_3d.append(point_3d.flatten())

    return np.array(points_3d)