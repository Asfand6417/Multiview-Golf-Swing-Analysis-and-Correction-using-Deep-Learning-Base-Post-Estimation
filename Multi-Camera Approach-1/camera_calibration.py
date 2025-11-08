import numpy as np
import cv2
from multicamera.camera_setup import setup_cameras

def calibrate_cameras():
    """Calibrate cameras using chessboard pattern"""
    # Define chessboard pattern size
    pattern_size = (9, 6)
    square_size = 0.025  # 2.5 cm squares

    # Prepare object points (3D points in real world space)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints1 = []  # 2D points in image plane for camera 1
    imgpoints2 = []  # 2D points in image plane for camera 2

    # Default image size in case we don't detect any chessboards
    img_size1 = (640, 480)  # Default fallback size
    img_size2 = (640, 480)  # Default fallback size

    cam1, cam2 = setup_cameras()

    # Capture multiple views of chessboard
    for _ in range(20):  # Capture 20 different views
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if not ret1 or not ret2:
            continue

        # Store image dimensions for later use
        img_size1 = (frame1.shape[1], frame1.shape[0])  # width, height
        img_size2 = (frame2.shape[1], frame2.shape[0])

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

        # If found in both cameras, add to calibration data
        if ret1 and ret2:
            # Refine corner detection
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            # Add to calibration data
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

            # Display detected corners
            cv2.drawChessboardCorners(frame1, pattern_size, corners1, ret1)
            cv2.drawChessboardCorners(frame2, pattern_size, corners2, ret2)
            cv2.imshow('Camera 1 Calibration', frame1)
            cv2.imshow('Camera 2 Calibration', frame2)

            # Wait for key press to continue
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    
    # Check if we found any chessboard patterns
    if len(objpoints) == 0:
        print("Error: No chessboard patterns were detected. Cannot calibrate cameras.")
        # Return a default calibration structure with identity matrices
        calibration_data = {
            'camera1_matrix': np.eye(3),
            'camera1_distortion': np.zeros(5),
            'camera2_matrix': np.eye(3),
            'camera2_distortion': np.zeros(5),
            'R': np.eye(3),  # Identity rotation
            'T': np.zeros((3, 1)),  # Zero translation
            'E': np.eye(3),  # Identity essential matrix
            'F': np.eye(3)   # Identity fundamental matrix
        }
        # Save calibration data
        np.save('calibration_data.npy', calibration_data)
        return calibration_data

    # Perform calibration for each camera
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, img_size1, None, None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, img_size2, None, None)

    # Perform stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret_stereo, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, img_size1,
        criteria=criteria_stereo, flags=flags
    )

    print("Stereo Calibration RMS Error:", ret_stereo)

    # Return camera matrices and transformation between cameras
    calibration_data = {
        'camera1_matrix': mtx1,
        'camera1_distortion': dist1,
        'camera2_matrix': mtx2,
        'camera2_distortion': dist2,
        'R': R,  # Rotation matrix between cameras
        'T': T,  # Translation vector between cameras
        'E': E,  # Essential matrix
        'F': F  # Fundamental matrix
    }

    # Save calibration data
    np.save('calibration_data.npy', calibration_data)

    return calibration_data
def test_calibration(calibration_data):
    """Test if calibration data appears valid"""
    required_keys = ['camera1_matrix', 'camera2_matrix', 'R', 'T']
    
    print("Testing calibration data...")
    
    # Check if all required keys exist
    for key in required_keys:
        if key not in calibration_data:
            print(f"ERROR: Missing key '{key}' in calibration data")
            return False
    
    # Check shapes
    mtx1 = calibration_data['camera1_matrix']
    mtx2 = calibration_data['camera2_matrix']
    R = calibration_data['R']
    T = calibration_data['T']
    
    if mtx1.shape != (3, 3):
        print(f"ERROR: Camera 1 matrix has incorrect shape: {mtx1.shape}, expected (3, 3)")
        return False
    
    if mtx2.shape != (3, 3):
        print(f"ERROR: Camera 2 matrix has incorrect shape: {mtx2.shape}, expected (3, 3)")
        return False
    
    if R.shape != (3, 3):
        print(f"ERROR: Rotation matrix has incorrect shape: {R.shape}, expected (3, 3)")
        return False
    
    if T.shape != (3, 1):
        print(f"ERROR: Translation vector has incorrect shape: {T.shape}, expected (3, 1)")
        return False
    
    # Check if matrices contain reasonable values
    if not np.all(np.isfinite(mtx1)) or not np.all(np.isfinite(mtx2)):
        print("ERROR: Camera matrices contain NaN or Inf values")
        return False
    
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(T)):
        print("ERROR: R or T contain NaN or Inf values")
        return False
    
    # Check if rotation matrix is valid (determinant should be close to 1)
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 0.1:
        print(f"WARNING: Rotation matrix may be invalid, determinant = {det_R}, expected close to 1.0")
    
    print("Calibration data appears valid")
    return True