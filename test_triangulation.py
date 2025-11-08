import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import from multi-camera-2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we need
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multi-camera-2'))
from reconstruction import triangulate_points, triangulate_sequence
from run_pipeline import extract_keypoints, load_calibration_data

def main():
    """Test the triangulation code with the problematic videos."""
    print("Testing triangulation with updated code...")

    # Load the same videos mentioned in the error message
    video1_path = "./../PoseVideos/17.mp4"  # Adjust path if needed
    video2_path = "./../PoseVideos/18.mp4"  # This was mentioned in the error

    # Check if the video files exist
    if not os.path.exists(video1_path):
        print(f"Error: Video file not found: {video1_path}")
        return

    if not os.path.exists(video2_path):
        print(f"Error: Video file not found: {video2_path}")
        return

    # Load calibration data
    calibration_path = "calibration_data.npz"  # Adjust path if needed
    if not os.path.exists(calibration_path):
        print(f"Error: Calibration data not found: {calibration_path}")
        return

    calibration_data = load_calibration_data(calibration_path)

    # Extract keypoints from both videos
    print(f"Extracting keypoints from {video1_path}")
    keypoints1 = extract_keypoints(video1_path)

    print(f"Extracting keypoints from {video2_path}")
    keypoints2 = extract_keypoints(video2_path)

    # Ensure both keypoint sequences have the same length
    min_length = min(len(keypoints1), len(keypoints2))
    keypoints1 = keypoints1[:min_length]
    keypoints2 = keypoints2[:min_length]

    print(f"Extracted keypoints from {len(keypoints1)} and {len(keypoints2)} frames")

    # Triangulate 3D points
    print("Triangulating 3D points...")
    landmarks_3d = triangulate_sequence(keypoints1, keypoints2, calibration_data)

    print(f"Successfully triangulated {len(landmarks_3d)} frames")
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
