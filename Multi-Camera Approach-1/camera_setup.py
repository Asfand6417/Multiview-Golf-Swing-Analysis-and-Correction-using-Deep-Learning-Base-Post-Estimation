import cv2

def setup_cameras(cam1_id=0, cam2_id=1):
    """
    Set up and return two camera objects for stereo vision.
    
    Args:
        cam1_id (int): ID of the first camera (default: 0)
        cam2_id (int): ID of the second camera (default: 1)
        
    Returns:
        tuple: Two camera objects (cv2.VideoCapture instances)
    """
    # Initialize the first camera
    cam1 = cv2.VideoCapture(cam1_id)
    if not cam1.isOpened():
        print(f"Camera ID {cam1_id} not available")
        # Fall back to video file if available
        cam1 = cv2.VideoCapture('camera1_recording.mp4')
    
    # Initialize the second camera
    cam2 = cv2.VideoCapture(cam2_id)
    if not cam2.isOpened():
        print(f"Camera ID {cam2_id} not available")
        # Fall back to video file if available
        cam2 = cv2.VideoCapture('camera2_recording.mp4')
    
    # Set camera properties (optional)
    for cam in [cam1, cam2]:
        if cam.isOpened():
            # Set resolution
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Set frame rate
            cam.set(cv2.CAP_PROP_FPS, 30)
    
    return cam1, cam2