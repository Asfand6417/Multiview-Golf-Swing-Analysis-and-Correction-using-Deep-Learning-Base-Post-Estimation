import cv2
import os
import time
from PoseModule import poseDetector
from utils.folder_management import create_video_folder
from utils.display_info import display_info
from utils.golf_rules import is_bad_swing
from utils.mpjpe_calculation import calculate_mpjpe

def process_video(video_path):
    output_dir = create_video_folder(video_path)
    cap = cv2.VideoCapture(video_path)
    detector = poseDetector()
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        h, w, c = img.shape
        phase = "Unknown"
        wrong_swing, mpjpe_score = False, None

        if lmList:
            # Placeholder for phase detection
            phase = "Top of Backswing"

            landmarks = [lm[:2] for lm in lmList]
            if is_bad_swing(landmarks, w, h):
                wrong_swing = True
                filename = os.path.join(output_dir, f"wrong_frame_{int(time.time())}.jpg")
                cv2.imwrite(filename, img)

            # Example MPJPE calculation (if ground truth available)
            # gt_coords = ...
            # mpjpe_score = calculate_mpjpe(landmarks, gt_coords)

        # Overlay info
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        img = display_info(img, fps, mpjpe=mpjpe_score, phase=phase, additional_text=["Golf Swing Analysis"])

        cv2.imshow("Video Analysis", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()