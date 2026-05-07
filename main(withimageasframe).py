import cv2
import mediapipe as mp
import time
import os
import math
import numpy as np
import csv
import json

# Create necessary directory structure
if not os.path.exists('utils'):
    os.makedirs('utils')
if not os.path.exists('ground_truth'):
    os.makedirs('ground_truth')

# Create CSV filename for results
csv_filename = 'golf_swing_data.csv'

# Define a class for pose detection and analysis
class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        
        # Initialize pose detector with proper named parameters
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        
        # Custom drawing specs
        self.connection_drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                # Draw green connections
                self.mpDraw.draw_landmarks(
                    img, 
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec,
                    connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        return img
        
    def findPosition(self, img, draw=False):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
        
    def checkGolfPose(self, img):
        """Check if golf pose is correct using multiple rules"""
        if len(self.lmList) < 33:  # Make sure we have enough landmarks
            return False, None
        
        # Initialize list to track incorrect body parts
        incorrect_parts = []
        
        # 1. Check right arm angle (shoulder-elbow-wrist)
        right_shoulder = self.lmList[12][1:]  # right shoulder
        right_elbow = self.lmList[14][1:]     # right elbow
        right_wrist = self.lmList[16][1:]     # right wrist
        
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        # Adjusted thresholds for golf swing - acceptable range is now 60-150 degrees
        if right_arm_angle < 60 or right_arm_angle > 150:
            cv2.line(img, (right_shoulder[0], right_shoulder[1]), (right_elbow[0], right_elbow[1]), (0, 0, 255), 4)
            cv2.line(img, (right_elbow[0], right_elbow[1]), (right_wrist[0], right_wrist[1]), (0, 0, 255), 4)
            cv2.putText(img, f"R-Arm: {int(right_arm_angle)}", (right_elbow[0], right_elbow[1] - 20), 
                      cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            incorrect_parts.append("right_arm")
        
        # 2. Check left arm angle (shoulder-elbow-wrist)
        left_shoulder = self.lmList[11][1:]  # left shoulder
        left_elbow = self.lmList[13][1:]     # left elbow
        left_wrist = self.lmList[15][1:]     # left wrist
        
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        # Adjusted thresholds - acceptable range is now 120-175 degrees for left arm (straighter)
        if left_arm_angle < 120 or left_arm_angle > 175:
            cv2.line(img, (left_shoulder[0], left_shoulder[1]), (left_elbow[0], left_elbow[1]), (0, 0, 255), 4)
            cv2.line(img, (left_elbow[0], left_elbow[1]), (left_wrist[0], left_wrist[1]), (0, 0, 255), 4)
            cv2.putText(img, f"L-Arm: {int(left_arm_angle)}", (left_elbow[0], left_elbow[1] - 20), 
                      cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            incorrect_parts.append("left_arm")
        
        # 3. Check torso angle (shoulders relative to hips)
        left_hip = self.lmList[23][1:]     # left hip
        right_hip = self.lmList[24][1:]    # right hip
        
        # Calculate the angle of the shoulder line relative to horizontal
        shoulder_angle = math.degrees(math.atan2(right_shoulder[1] - left_shoulder[1], 
                                             right_shoulder[0] - left_shoulder[0]))
        
        # Calculate the angle of the hip line relative to horizontal
        hip_angle = math.degrees(math.atan2(right_hip[1] - left_hip[1], 
                                         right_hip[0] - left_hip[0]))
        
        # Calculate torso rotation (difference between shoulder and hip alignment)
        torso_rotation = abs(shoulder_angle - hip_angle)
        
        # For golf backswing, we want significant rotation (30-90 degrees)
        if torso_rotation < 20 or torso_rotation > 100:
            # Draw torso in red
            cv2.line(img, (left_shoulder[0], left_shoulder[1]), (right_shoulder[0], right_shoulder[1]), (0, 0, 255), 4)
            cv2.line(img, (left_hip[0], left_hip[1]), (right_hip[0], right_hip[1]), (0, 0, 255), 4)
            cv2.line(img, (left_shoulder[0], left_shoulder[1]), (left_hip[0], left_hip[1]), (0, 0, 255), 4)
            cv2.line(img, (right_shoulder[0], right_shoulder[1]), (right_hip[0], right_hip[1]), (0, 0, 255), 4)
            cv2.putText(img, f"Rot: {int(torso_rotation)}", ((left_hip[0] + right_hip[0])//2, 
                                                        (left_hip[1] + right_hip[1])//2 - 20), 
                      cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            incorrect_parts.append("torso_rotation")
            
        # 4. Check knee bend (for both knees)
        left_hip = self.lmList[23][1:]     # left hip
        left_knee = self.lmList[25][1:]    # left knee
        left_ankle = self.lmList[27][1:]   # left ankle
        
        right_hip = self.lmList[24][1:]    # right hip
        right_knee = self.lmList[26][1:]   # right knee
        right_ankle = self.lmList[28][1:]  # right ankle
        
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # For golf stance, knees should be slightly bent (150-175 degrees)
        if left_knee_angle < 150 or left_knee_angle > 175:
            cv2.line(img, (left_hip[0], left_hip[1]), (left_knee[0], left_knee[1]), (0, 0, 255), 4)
            cv2.line(img, (left_knee[0], left_knee[1]), (left_ankle[0], left_ankle[1]), (0, 0, 255), 4)
            cv2.putText(img, f"L-Knee: {int(left_knee_angle)}", (left_knee[0], left_knee[1] - 20), 
                      cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            incorrect_parts.append("left_knee")
            
        if right_knee_angle < 150 or right_knee_angle > 175:
            cv2.line(img, (right_hip[0], right_hip[1]), (right_knee[0], right_knee[1]), (0, 0, 255), 4)
            cv2.line(img, (right_knee[0], right_knee[1]), (right_ankle[0], right_ankle[1]), (0, 0, 255), 4)
            cv2.putText(img, f"R-Knee: {int(right_knee_angle)}", (right_knee[0], right_knee[1] - 20), 
                      cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            incorrect_parts.append("right_knee")
        
        # Return True if any part is incorrect, with the list of incorrect parts
        return len(incorrect_parts) > 0, incorrect_parts
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angle_degrees = np.degrees(angle)
        
        return angle_degrees

def display_info(img, fps, wrong_pose=False, incorrect_parts=None, phase="Unknown", mpjpe=None):
    """Display information on the image"""
    # Display FPS
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display pose status
    status_color = (0, 0, 255) if wrong_pose else (0, 255, 0)
    status_text = "BAD POSE" if wrong_pose else "GOOD POSE"
    cv2.putText(img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Display incorrect parts if any
    if incorrect_parts and len(incorrect_parts) > 0:
        issues = ", ".join(incorrect_parts)
        cv2.putText(img, f"Issues: {issues}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display swing phase
    cv2.putText(img, f"Phase: {phase}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Show MPJPE if available
    if mpjpe is not None:
        mpjpe_color = (0, 255, 255)  # Yellow by default
        if mpjpe < 20:  # Good match with ground truth
            mpjpe_color = (0, 255, 0)  # Green
        elif mpjpe > 50:  # Poor match with ground truth
            mpjpe_color = (0, 0, 255)  # Red
            
        cv2.putText(img, f"MPJPE: {mpjpe:.2f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mpjpe_color, 2)
    
    # Add title
    cv2.putText(img, "MULTI-VIEW GOLF SWING ANALYSIS AND CORRECTION", (img.shape[1]//2 - 300, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return img

def load_ground_truth(video_name):
    """Load the ground truth data for a specific video
    
    This function tries to load from a JSON file containing expert landmark positions
    """
    # Expected format: ground_truth/video_name.json
    gt_file = os.path.join('ground_truth', f"{video_name}.json")
    
    # Check if ground truth file exists
    if not os.path.exists(gt_file):
        # If not, create an example file structure to be filled in later
        create_sample_ground_truth(gt_file)
        print(f"Created sample ground truth file at {gt_file}")
        print("Please replace with actual expert landmark data.")
        return None
        
    # Load the ground truth data
    try:
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)
            
        # Extract landmarks from the loaded data
        landmarks = gt_data.get('landmarks', [])
        if not landmarks:
            print(f"Warning: No landmarks found in {gt_file}")
            return None
            
        return landmarks
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        return None

def create_sample_ground_truth(file_path):
    """Create a sample ground truth file structure"""
    # Create a minimal example structure
    sample_data = {
        "description": "Expert golf swing landmark positions",
        "landmarks": [
            # Format: Each frame has 33 landmarks with x,y coordinates
            # Frame 1 example (just sample values - replace with real data)
            [
                [320, 240],  # landmark 0 position
                [325, 245],  # landmark 1 position
                # ... add all 33 landmarks
            ]
            # Add more frames as needed
        ]
    }
    
    # Save the sample file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(sample_data, f, indent=2)

def calculate_mpjpe(pred_coords, gt_coords):
    """Calculate Mean Per Joint Position Error between predicted and ground truth coordinates"""
    # If inputs are None or empty, return None
    if pred_coords is None or gt_coords is None or len(pred_coords) == 0 or len(gt_coords) == 0:
        return None
        
    # Format the coordinate arrays to just x,y positions
    if len(pred_coords[0]) > 2:  # Handle format [id, x, y]
        pred = np.array([[p[1], p[2]] for p in pred_coords])
    else:
        pred = np.array(pred_coords)
        
    gt = np.array(gt_coords)
    
    # Make sure shapes match - truncate if necessary
    min_length = min(len(pred), len(gt))
    pred = pred[:min_length]
    gt = gt[:min_length]
    
    # Calculate Euclidean distance for each joint
    distances = np.linalg.norm(pred - gt, axis=1)
    
    # Return mean distance across all joints
    return np.mean(distances)

def detect_swing_phase(frame_num, landmarks):
    """Detect the current golf swing phase based on pose and frame number"""
    # This is a simplified logic - in a real application, you would need more sophisticated detection
    # Phases: Address, Takeaway, Backswing, Top, Downswing, Impact, Follow-Through
    
    # For simplicity, we'll just divide the swing into phases based on frame number
    # In a real application, you would analyze the actual pose
    if frame_num < 10:
        return "Address"
    elif frame_num < 20:
        return "Takeaway"
    elif frame_num < 30:
        return "Backswing"
    elif frame_num < 35:
        return "Top"
    elif frame_num < 45:
        return "Downswing"
    elif frame_num < 50:
        return "Impact"
    else:
        return "Follow-Through"

def process_video():
    """Main function to process a video and perform pose analysis"""
    # Video path 
    video_path = 'PoseVideos/42.mp4'
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        if not os.path.exists('PoseVideos'):
            os.makedirs('PoseVideos')
            print(f"Created directory 'PoseVideos'. Please place your video files there.")
        return
    
    # Extract video name for folder creation and ground truth loading
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output directory for wrong frames
    output_dir = os.path.join("Wrong_Frames", video_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Load ground truth data
    ground_truth = load_ground_truth(video_name)
    if ground_truth:
        print(f"Loaded ground truth data with {len(ground_truth)} frames")
    else:
        print("No ground truth data available. MPJPE calculation will be skipped.")
    
    # Initialize CSV file with headers if it doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Video Name', 'Frame', 'Timestamp', 'MPJPE', 'Alignment Error', 'Phase', 'Incorrect Parts', 'FPS'])
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    # Initialize detector
    detector = poseDetector()
    
    # Time tracking
    pTime = 0
    frame_count = 0
    wrong_frames_count = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Detect pose
        img = detector.findPose(img)
        
        # Find positions of landmarks
        lmList = detector.findPosition(img)
        
        # Variables for CSV data
        mpjpe_score = None
        is_wrong_pose = False
        incorrect_parts = None
        
        # Detect current phase based on frame number and pose
        current_phase = detect_swing_phase(frame_count, lmList)

        if len(lmList) > 0:
            # Check golf pose
            is_wrong_pose, incorrect_parts = detector.checkGolfPose(img)

            # Calculate MPJPE if ground truth is available
            if ground_truth and frame_count <= len(ground_truth):
                # Use the corresponding frame from ground truth
                gt_frame = ground_truth[frame_count - 1]
                mpjpe_score = calculate_mpjpe(lmList, gt_frame)

            # If pose is wrong, save the frame
            if is_wrong_pose:
                filename = os.path.join(output_dir, f"wrong_frame_{frame_count}.jpg")
                cv2.imwrite(filename, img)
                wrong_frames_count += 1
                print(f"Saved frame {frame_count} to {filename}")

            # Display information
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display information on the frame
        img = display_info(img, fps, is_wrong_pose, incorrect_parts, current_phase, mpjpe_score)

        # Show image
        cv2.imshow("MULTI-VIEW GOLF SWING ANALYSIS AND CORRECTION", img)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write data to CSV
        timestamp_str = f"{frame_count / 30:.2f}"  # Estimate timestamp based on frame count and 30fps
        video_name = os.path.basename(video_path)
        mpjpe_value = f"{mpjpe_score:.4f}" if mpjpe_score is not None else ''
        alignment_error_value = '0.0' if not is_wrong_pose else '1.0'  # Simple binary for alignment error
        incorrect_parts_str = ','.join(incorrect_parts) if incorrect_parts else ''

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([video_name, frame_count, timestamp_str, mpjpe_value,
                             alignment_error_value, current_phase, incorrect_parts_str, int(fps)])

    print(f"Processed {frame_count} frames")
    print(f"Found {wrong_frames_count} frames with incorrect poses")
    print(f"Results saved to {csv_filename}")
    print(f"Wrong frames saved to {output_dir}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()
