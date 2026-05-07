# Main Module Documentation

The main module (`main.py`) is the core component of the GolfPose system, providing functionality for 3D pose analysis, swing phase detection, and evaluation metrics.

## Classes

### DirectLinearTransform

A class that implements Direct Linear Transformation for trajectory smoothing.

#### Methods

- `__init__(window_size=5)`: Initialize DLT smoother with specified window size
- `smooth_trajectory(landmarks_sequence)`: Apply DLT to smooth a sequence of landmarks
- `_apply_dlt_to_point(point_trajectory)`: Apply DLT to a single point's trajectory

#### Example

```python
from main import DirectLinearTransform

# Initialize smoother with window size of 7
smoother = DirectLinearTransform(window_size=7)

# Smooth a sequence of landmarks
smoothed_landmarks = smoother.smooth_trajectory(landmarks_sequence)
```

### SwingPhaseSegmenter

A class that implements golf swing phase segmentation using Support Vector Machine (SVM).

#### Swing Phases

The segmenter identifies the following phases:
- Address: Initial setup position
- Takeaway: Initial movement of the club
- Backswing: Movement of the club away from the target
- Top: Top of the swing
- Downswing: Movement of the club toward the target
- Impact: Club contact with the ball
- Follow-through: Completion of the swing

#### Methods

- `__init__(model_path=None)`: Initialize the swing phase segmenter
- `extract_features(landmarks)`: Extract features from pose landmarks
- `_calculate_angle(a, b, c)`: Calculate angle between three 3D points
- `_angle_between_vectors(v1, v2)`: Calculate angle between two vectors
- `train(landmark_sequences, labels)`: Train the SVM model on labeled sequences
- `predict(landmarks)`: Predict the swing phase for a single frame
- `save_model(model_path)`: Save the trained model to disk

#### Example

```python
from main import SwingPhaseSegmenter

# Initialize segmenter
segmenter = SwingPhaseSegmenter()

# Train the model with labeled data
segmenter.train(landmark_sequences, phase_labels)

# Save the trained model
segmenter.save_model("models/swing_phase_svm.pkl")

# Predict phase for new landmarks
phase = segmenter.predict(landmarks)
print(f"Current swing phase: {phase}")
```

### PoseEvaluator

A class that provides evaluation metrics for pose estimation.

#### Methods

- `__init__()`: Initialize the pose evaluator
- `calculate_mpjpe(pred_coords, gt_coords)`: Calculate Mean Per Joint Position Error
- `evaluate_phase_accuracy(predicted_phases, ground_truth_phases)`: Evaluate phase detection accuracy
- `add_subjective_feedback(score, comments="")`: Add subjective feedback from a coach
- `generate_report()`: Generate a comprehensive evaluation report

#### Example

```python
from main import PoseEvaluator

# Initialize evaluator
evaluator = PoseEvaluator()

# Calculate MPJPE between predicted and ground truth coordinates
mpjpe = evaluator.calculate_mpjpe(predicted_landmarks, ground_truth_landmarks)

# Evaluate phase detection accuracy
accuracy, recall = evaluator.evaluate_phase_accuracy(predicted_phases, ground_truth_phases)

# Add subjective feedback
evaluator.add_subjective_feedback(score=8, comments="Good swing, but slight overrotation")

# Generate report
report = evaluator.generate_report()
print(f"Average MPJPE: {report['mpjpe']['mean']}")
```

### Pose3DVisualizer

A class that visualizes 3D pose data and creates animations.

#### Methods

- `__init__()`: Initialize the MediaPipe pose detector and visualization settings
- `process_video(video_path, ground_truth_path=None)`: Process video file and extract 3D landmarks
- `_load_ground_truth(gt_path)`: Load ground truth data from file
- `_reorient_landmarks(landmarks)`: Reorient landmarks for proper floor alignment
- `_get_pose_connections()`: Get the joint connections used by MediaPipe Pose
- `_update_plot(frame_idx)`: Update the 3D plot for a specific frame
- `animate(show=True, save=False, output_path='output_3d_skeleton.mp4')`: Create an animation of the 3D pose
- `train_phase_segmenter(training_data_path)`: Train the SVM phase segmenter

#### Example

```python
from main import Pose3DVisualizer

# Initialize visualizer
visualizer = Pose3DVisualizer()

# Process a video
visualizer.process_video("PoseVideos/golf_swing.mp4", "ground_truth/gt_data.json")

# Create and save animation
visualizer.animate(show=True, save=True, output_path="results/3d_visualization.mp4")
```

## Main Function

The `main()` function demonstrates how to use the Pose3DVisualizer class:

1. Sets up the visualizer
2. Checks for training data and trains the SVM if necessary
3. Processes a video file
4. Adds subjective feedback
5. Creates and displays the animation

## Usage Notes

- The system requires MediaPipe for pose detection
- For best results, ensure the video has good lighting and the subject is clearly visible
- Ground truth data is optional but can be used for evaluation
- The system can process videos of various resolutions and frame rates