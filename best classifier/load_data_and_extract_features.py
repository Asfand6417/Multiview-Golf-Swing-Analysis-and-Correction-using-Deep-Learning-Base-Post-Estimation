import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assume you have a function to extract pose features (e.g., from MediaPipe)
# for a given video ID and frame range.
# def extract_pose_features_for_video_segment(video_id, start_frame, end_frame, video_base_path):
#     # This function would:
#     # 1. Construct the video file path (e.g., video_base_path + f"{video_id}.mp4")
#     # 2. Open the video, seek to start_frame.
#     # 3. Process frames from start_frame to end_frame using MediaPipe (or your pose estimator).
#     # 4. For each frame, extract keypoints (e.g., 33 landmarks * 2 coords = 66 features, or 33*3 for 3D).
#     # 5. Aggregate features for the segment (e.g., average pose, or flatten all poses in the window).
#     # For simplicity, let's imagine it returns a single feature vector for the segment.
#     # In reality, you might use LSTMs or Transformers for sequences of poses.
#     # Placeholder:
#     return np.random.rand(1, 66) # Example: 33 2D keypoints flattened

# --- Define phase mapping more concretely ---
# This mapping is illustrative and needs careful thought based on the 'events' definition.
# Let's simplify to fewer phases for a start.
# P1 (Address): E0-E1
# P2 (Backswing): E1-E2
# P3 (Downswing): E2-E5 (assuming E3 is start of downswing near E2)
# P4 (Follow-Through): E5-E7
# P5 (Finish): E7-E9

phase_definitions = {
    "Address": (0, 1),  # Uses events[0] to events[1]
    "Backswing": (1, 2),
    "Downswing": (2, 5),  # Assuming events[2] is Top, events[5] is Impact
    "FollowThrough": (5, 7),
    "Finish": (7, 9)  # Using 9 as end of sequence. events[8] might be start of idle.
}
phase_labels_map = {name: i for i, name in enumerate(phase_definitions.keys())}

df = pd.read_csv('.\..\GolfDB\GolfDB.csv')
all_features = []
all_labels = []

# You'll need the actual video files corresponding to the IDs in GolfDB.csv
# Let's assume they are in a folder "./../PoseVideos/" and named like "0.mp4", "1.mp4", etc.
# based on the 'id' column. (This needs verification based on your actual video naming)
VIDEO_BASE_PATH = "./../GolfDB/videos_160/videos_160"

# For simplicity, let's take one representative feature vector per defined phase segment
# In a real system, you'd sample many frames or windows from each phase.
for index, row in df.iterrows():
    video_id = row['id']
    try:
        # The 'events' column is a string representation of a list
        events_str = row['events'].replace('[', '').replace(']', '').replace('\n', ' ')
        events = np.fromstring(events_str, sep=',', dtype=int)
        if len(events) < 10:  # Check if events array is complete
            print(f"Skipping video {video_id}: Incomplete events array {events}")
            continue
    except Exception as e:
        print(f"Skipping video {video_id} due to error parsing events: {e}, events_str: '{row['events']}'")
        continue

    video_file_path = os.path.join(VIDEO_BASE_PATH, f"{video_id}.mp4")  # Example path
    if not os.path.exists(video_file_path):
        # print(f"Video file not found for ID {video_id}: {video_file_path}. Skipping.")
        continue  # Skip if video file doesn't exist

    for phase_name, (start_event_idx, end_event_idx) in phase_definitions.items():
        start_frame = events[start_event_idx]
        end_frame = events[end_event_idx]

        if start_frame >= end_frame:  # Skip if segment is empty or invalid
            # print(f"Skipping phase {phase_name} for video {video_id}: start_frame {start_frame} >= end_frame {end_frame}")
            continue

        # In a real scenario, you'd extract features from frames in this range.
        # For this example, we'll generate placeholder features.
        # You would call your actual feature extraction function here.
        # segment_features = extract_pose_features_for_video_segment(video_id, start_frame, end_frame, VIDEO_BASE_PATH)

        # Placeholder: generate random features for each phase of each video
        # The dimensionality (e.g., 198 for 33 3D landmarks) depends on your feature extraction
        # For example, if using average 3D pose (33 landmarks * 3 coords)
        num_pose_features = 33 * 3  # Example for 3D world landmarks
        segment_features = np.random.rand(num_pose_features)

        all_features.append(segment_features)
        all_labels.append(phase_labels_map[phase_name])

if not all_features:
    print("No features extracted. Check video paths, event parsing, and feature extraction logic.")
    exit()

X = np.array(all_features)
y = np.array(all_labels)

print(f"Total samples extracted: {X.shape[0]}")
if X.shape[0] == 0:
    print("Stopping: No data samples were generated.")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                    stratify=y if len(np.unique(y)) > 1 else None)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)