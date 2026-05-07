import cv2
import mediapipe as mp
import time
import os
import math  # Added the missing import

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

pose = mpPose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture('PoseVideos/42.mp4')
output_dir = "wrong_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pTime = 0
frame_count = 0

def is_segment_wrong(landmarks, w, h):
    # Example: Check elbow angle (landmarks 11, 13, 15 for left arm)
    # and return True if wrong, False if correct
    # Repeat similarly for other segments as needed
    def get_angle(p1, p2, p3):
        x1, y1 = landmarks[p1].x * w, landmarks[p1].y * h
        x2, y2 = landmarks[p2].x * w, landmarks[p2].y * h
        x3, y3 = landmarks[p3].x * w, landmarks[p3].y * h
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        return angle if angle >= 0 else angle + 360

    # Example threshold for elbow angle (e.g., should be between 45 and 135 degrees)
    elbow_angle = get_angle(11, 13, 15)  # Left arm, for instance
    if elbow_angle < 45 or elbow_angle > 135:
        return True  # Wrong segment

    # Add additional checks for other segments as needed

    return False

while True:
    success, img = cap.read()
    if not success:
        break
    frame_count += 1
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    h, w, c = img.shape
    wrong_segment = False

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Check for wrong segments
        if is_segment_wrong(results.pose_landmarks.landmark, w, h):
            wrong_segment = True

        # Draw segments in red if wrong
        if wrong_segment:
            # Example: draw line between shoulder and elbow in red
            LM = results.pose_landmarks.landmark
            points = {
                'left_shoulder': LM[11],
                'left_elbow': LM[13]
            }
            cx1, cy1 = int(points['left_shoulder'].x * w), int(points['left_shoulder'].y * h)
            cx2, cy2 = int(points['left_elbow'].x * w), int(points['left_elbow'].y * h)
            cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 0, 255), 4)  # Red line

            # Save frame
            filename = os.path.join(output_dir, f"wrong_frame_{frame_count}.jpg")
            cv2.imwrite(filename, img)
        else:
            # Draw normally
            pass

        # Optionally: draw all keypoints
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)