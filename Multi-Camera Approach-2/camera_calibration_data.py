import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load calibration data
calibration_data = np.load('calibration_data.npz', allow_pickle=True)

# Print calibration parameters
print("Camera 1 Matrix:")
print(calibration_data['camera1_matrix'])
print("\nCamera 2 Matrix:")
print(calibration_data['camera2_matrix'])
print("\nRotation Matrix:")
print(calibration_data['R'])
print("\nTranslation Vector:")
print(calibration_data['T'])

# Load a frame from each video
video1_path = "./../PoseVideos/17.mp4"
video2_path = "./../PoseVideos/18.mp4"

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

cap1.release()
cap2.release()

# Display frames
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
plt.title("Camera 1")
plt.subplot(122)
plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
plt.title("Camera 2")
plt.savefig("camera_views.png")
plt.show()
import cv2

# Check video resolutions
video1 = cv2.VideoCapture("./../PoseVideos/17.mp4")
width1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

video2 = cv2.VideoCapture("./../PoseVideos/18.mp4")
width2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))

video1.release()
video2.release()

print(f"Camera 1: {width1}x{height1}")
print(f"Camera 2: {width2}x{height2}")

# Load calibration to check camera matrices
calibration = np.load("calibration_data.npz", allow_pickle=True)
cam1_matrix = calibration["camera1_matrix"]
cam2_matrix = calibration["camera2_matrix"]

# Principal points should be approximately half the resolution
print(f"Camera 1 principal point: ({cam1_matrix[0,2]}, {cam1_matrix[1,2]})")
print(f"Camera 2 principal point: ({cam2_matrix[0,2]}, {cam2_matrix[1,2]})")