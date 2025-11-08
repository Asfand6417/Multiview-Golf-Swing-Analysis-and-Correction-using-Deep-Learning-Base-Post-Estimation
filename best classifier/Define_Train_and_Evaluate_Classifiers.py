# --- START OF COMBINED SCRIPT ---
import pandas as pd
import numpy as np
import os
import cv2  # Added for video processing
import mediapipe as mp  # Added for pose estimation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# --- Part 1: Load Data and Extract Features/Labels ---

# --- NEW/CORRECTED FEATURE EXTRACTION FUNCTION ---
def extract_pose_features_for_video_segment(video_file_path, start_frame, end_frame, pose_estimator):
    """
    Extracts and aggregates pose features for a specific segment of a video file using MediaPipe.
    It returns a single feature vector representing the average pose over the segment.
    """
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        # print(f"    Warning: Could not open video {video_file_path}")
        return None

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    collected_poses = []

    # Process frames from start_frame to end_frame
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends or there's an error

        # Convert the BGR image to RGB as MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and get pose landmarks
        results = pose_estimator.process(frame_rgb)

        # We use pose_world_landmarks for 3D coordinates, which are better for biomechanics
        if results.pose_world_landmarks:
            # Flatten the 33 landmarks (x, y, z) into a single 99-element array
            pose_3d = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]).flatten()
            collected_poses.append(pose_3d)

    cap.release()

    if not collected_poses:
        # This can happen if no poses were detected in any frame of the segment
        return None

    # Aggregate features for the segment. Taking the mean is a robust way to get an "average pose".
    aggregated_features = np.mean(collected_poses, axis=0)
    return aggregated_features


# Define phase mapping
phase_definitions = {
    "Address": (0, 1),
    "Backswing": (1, 2),
    "Downswing": (2, 5),
    "FollowThrough": (5, 7),
    "Finish": (7, 8)
}
phase_labels_map = {name: i for i, name in enumerate(phase_definitions.keys())}

print("Loading GolfDB.csv...")
try:
    df = pd.read_csv('.\..\GolfDB\GolfDB.csv')
except FileNotFoundError:
    print("ERROR: GolfDB.csv not found. Please make sure it's in the correct directory.")
    exit()

all_features = []
all_labels = []
VIDEO_BASE_PATH = "./../GolfDB/videos_160/videos_160"

print("Extracting REAL features using MediaPipe (this will take some time)...")

# --- INITIALIZE MEDIAPIPE POSE ESTIMATOR ONCE ---
mp_pose = mp.solutions.pose
with mp_pose.Pose(
        static_image_mode=False,  # False for video, as it tracks between frames
        model_complexity=2,  # Use 2 for highest accuracy
        enable_segmentation=False,  # Not needed for this task
        min_detection_confidence=0.5
) as pose_estimator:
    for index, row in df.iterrows():
        video_id = row['id']
        try:
            events_str = str(row['events']).replace('\n', ' ')
            events_str_cleaned = ''.join(c for c in events_str if c.isdigit() or c == ',' or c == ' ' or c == '.')
            events = np.fromstring(events_str_cleaned, sep=',', dtype=int)

            if len(events) < max(max(v) for v in phase_definitions.values()) + 1:
                continue
        except Exception as e:
            continue

        video_file_path = os.path.join(VIDEO_BASE_PATH, f"{video_id}.mp4")
        if not os.path.exists(video_file_path):
            continue

        for phase_name, (start_event_idx, end_event_idx) in phase_definitions.items():
            if start_event_idx >= len(events) or end_event_idx >= len(events):
                continue

            start_frame = events[start_event_idx]
            end_frame = events[end_event_idx]

            if start_frame >= end_frame:
                continue

            # --- CALL THE CORRECTED FEATURE EXTRACTION FUNCTION ---
            segment_features = extract_pose_features_for_video_segment(
                video_file_path, start_frame, end_frame, pose_estimator
            )

            if segment_features is not None:
                all_features.append(segment_features)
                all_labels.append(phase_labels_map[phase_name])

# --- The rest of the script remains largely the same ---

if not all_features:
    print("CRITICAL ERROR: No features were extracted. Check video paths and MediaPipe functionality.")
    exit()

X = np.array(all_features)
y = np.array(all_labels)

print(f"\nTotal samples extracted: {X.shape[0]}")
if X.shape[0] == 0:
    print("Stopping: No data samples were generated.")
    exit()

stratify_option = y if len(np.unique(y)) > 1 else None

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=stratify_option)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Data prepared: X_train_scaled shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")

# --- Part 2: Define, Train, and Evaluate Classifiers ---
classifiers = {
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
output_plots_dir = "classifier_results"
os.makedirs(output_plots_dir, exist_ok=True)

print("\n--- Training and Evaluating Classifiers ---")
if X_train_scaled.shape[0] == 0:
    print("ERROR: Training data is empty.")
    exit()

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    try:
        if name == "KNN":
            min_samples_in_train = np.min(np.unique(y_train, return_counts=True)[1]) if len(y_train) > 0 else 0
            n_neighbors = min(5, max(1, min_samples_in_train))
            if n_neighbors == 0 and X_train_scaled.shape[0] > 0: n_neighbors = 1
            if n_neighbors == 0:
                print(f"  Skipping KNN as training set is too small.")
                continue
            clf.n_neighbors = n_neighbors
            print(f"  (Using n_neighbors={n_neighbors} for KNN)")

        clf.fit(X_train_scaled, y_train)

        if X_test_scaled.shape[0] > 0:
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            unique_labels_in_data = sorted(list(np.unique(np.concatenate((y_train, y_test)))))
            current_target_names = [key for key, val in phase_labels_map.items() if val in unique_labels_in_data]

            report = classification_report(y_test, y_pred, labels=unique_labels_in_data,
                                           target_names=current_target_names, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels_in_data)

            results[name] = {"accuracy": accuracy, "report": report, "cm": cm, "model": clf}

            print(f"--- {name} Results ---")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)

            plt.figure(figsize=(max(6, len(current_target_names) * 1.2), max(5, len(current_target_names))))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=current_target_names,
                        ytickicklabels=current_target_names)
            plt.title(f'Confusion Matrix - {name} (Acc: {accuracy:.2f})')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(output_plots_dir, f"cm_{name.replace(' ', '_')}.png"))
            plt.close()
        else:
            results[name] = {"accuracy": "N/A (No test data)", "report": "N/A", "cm": None, "model": clf}
    except Exception as e:
        print(f"ERROR during training/evaluation of {name}: {e}")
        results[name] = {"accuracy": "Error", "report": str(e), "cm": None, "model": None}

# --- Comparing Results ---
print("\n\n--- Overall Comparison ---")
best_classifier_name = ""
best_accuracy = -1.0
for name, res in results.items():
    if isinstance(res['accuracy'], float):
        print(f"Classifier: {name}, Accuracy: {res['accuracy']:.4f}")
        if res['accuracy'] > best_accuracy:
            best_accuracy = res['accuracy']
            best_classifier_name = name
    else:
        print(f"Classifier: {name}, Accuracy: {res['accuracy']}")

if best_classifier_name:
    print(f"\nBest performing classifier: {best_classifier_name} with accuracy: {best_accuracy:.4f}")
else:
    print("\nCould not determine best classifier.")

print(f"\nConfusion matrix plots saved to '{output_plots_dir}' directory.")
# --- END OF COMBINED SCRIPT ---