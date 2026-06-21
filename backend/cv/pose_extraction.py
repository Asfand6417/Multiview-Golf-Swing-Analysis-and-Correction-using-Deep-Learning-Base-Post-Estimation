import cv2
import mediapipe as mp


POSE_JOINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def extract_pose_keypoints(video_path, visibility_threshold=0.7):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    mp_pose = mp.solutions.pose
    frames = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break

            height, width = image.shape[:2]
            timestamp = frame_index / fps if fps > 0 else float(frame_index)
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            points = {name: None for name in POSE_JOINT_NAMES}
            visibility = {name: None for name in POSE_JOINT_NAMES}

            if results.pose_landmarks:
                for joint_index, landmark in enumerate(results.pose_landmarks.landmark):
                    if joint_index >= len(POSE_JOINT_NAMES):
                        continue
                    name = POSE_JOINT_NAMES[joint_index]
                    visibility[name] = float(landmark.visibility)
                    if landmark.visibility >= visibility_threshold:
                        points[name] = [float(landmark.x * width), float(landmark.y * height)]

            frames.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "points": points,
                "visibility": visibility,
            })
            frame_index += 1

    cap.release()
    return {
        "fps": fps,
        "frame_count": len(frames) if frames else reported_frame_count,
        "joint_names": POSE_JOINT_NAMES,
        "frames": frames,
    }
