import mediapipe as mp
import cv2
import numpy as np

def extract_keypoints(frame):
    """Extract keypoints from a single frame using MediaPipe"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = pose.process(rgb_frame)
    keypoints = []
    confidences = []

    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # Convert to pixel coordinates
            h, w, _ = frame.shape
            keypoint = [landmark.x * w, landmark.y * h]
            confidence = landmark.visibility
            keypoints.append(keypoint)
            confidences.append(confidence)

            # Draw keypoints for visualization
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 5, (0, 255, 0), -1)

    # Clean up resources
    pose.close()

    return np.array(keypoints), np.array(confidences), frame