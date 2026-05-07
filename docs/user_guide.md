# GolfPose User Guide

This guide provides step-by-step instructions for using the GolfPose system to analyze golf swings. It covers installation, basic usage, and advanced features.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Single Camera Analysis](#single-camera-analysis)
4. [Multi-Camera Analysis](#multi-camera-analysis)
5. [Interpreting Results](#interpreting-results)
6. [Troubleshooting](#troubleshooting)

## Installation

### System Requirements

- Windows, macOS, or Linux operating system
- Python 3.8 or higher
- Webcam or video files of golf swings
- For multi-camera setup: two or more synchronized cameras

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/golfpose.git
cd golfpose
```

### Step 2: Create a Virtual Environment

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

For a quick demonstration of GolfPose's capabilities:

1. Ensure you have a golf swing video in the `PoseVideos` directory
2. Run the main script:

```bash
python main.py
```

This will:
- Process the default video (or the first video found in the PoseVideos directory)
- Generate a 3D visualization of the golf swing
- Display the animation and save it to the results directory

## Single Camera Analysis

### Preparing Your Video

For best results:
1. Record the golf swing from a side view
2. Ensure the golfer's full body is visible throughout the swing
3. Use good lighting conditions
4. Record at 30fps or higher
5. Place the video in the `PoseVideos` directory

### Running the Analysis

#### Basic Analysis

```python
from main import Pose3DVisualizer

# Initialize the visualizer
visualizer = Pose3DVisualizer()

# Process your video
visualizer.process_video("PoseVideos/your_video.mp4")

# Create and display the animation
visualizer.animate(show=True, save=True, output_path="results/your_analysis.mp4")
```

#### Advanced Analysis with Ground Truth Data

If you have ground truth data (e.g., from a professional motion capture system):

```python
visualizer.process_video(
    "PoseVideos/your_video.mp4",
    ground_truth_path="ground_truth/your_data.json"
)

# Generate a report
report = visualizer.evaluator.generate_report()
print(f"MPJPE: {report['mpjpe']['mean']:.4f}")
```

### Customizing Visualization

You can customize the 3D visualization by modifying parameters in the `_update_plot` method:

```python
# Change the view angle for better visualization
self.ax.view_init(elev=20, azim=45)

# Adjust axis limits for different scaling
self.ax.set_xlim(-1, 1)
self.ax.set_ylim(-1, 1)
self.ax.set_zlim(0, 2)
```

## Multi-Camera Analysis

### Camera Setup

For multi-camera analysis:
1. Position two cameras approximately 90 degrees apart
2. Ensure both cameras can see the golfer's full body
3. Start recording on both cameras simultaneously
4. Use the same frame rate for both cameras

### Camera Calibration

For accurate 3D reconstruction, calibrate your cameras:

1. Print a checkerboard pattern
2. Record the pattern from both cameras simultaneously
3. Run the calibration script:

```python
from multicamera.camera_calibration import calibrate_cameras, save_calibration_data

# Calibrate cameras using images in the calibration directory
calibration_data = calibrate_cameras("calibration_images/")

# Save calibration data for future use
save_calibration_data(calibration_data, "calibration_data.npy")
```

### Running Multi-Camera Analysis

```python
from multicamera.media_pipe_3D_landmark import process_dual_videos_with_single_3d_skeleton

# Process videos from two camera angles
process_dual_videos_with_single_3d_skeleton(
    "PoseVideos/camera1_swing.mp4",
    "PoseVideos/camera2_swing.mp4",
    output_dir="output/dual_camera_results"
)
```

## Interpreting Results

### Swing Phase Detection

GolfPose automatically segments the golf swing into phases:
- **Address**: Initial setup position
- **Takeaway**: Initial movement of the club
- **Backswing**: Movement of the club away from the target
- **Top**: Top of the swing
- **Downswing**: Movement of the club toward the target
- **Impact**: Club contact with the ball
- **Follow-through**: Completion of the swing

The current phase is displayed in the visualization and included in the output data.

### Evaluation Metrics

#### MPJPE (Mean Per Joint Position Error)

MPJPE measures the average Euclidean distance between predicted joint positions and ground truth positions. Lower values indicate better accuracy.

Typical values:
- < 50mm: Excellent accuracy
- 50-100mm: Good accuracy
- > 100mm: Poor accuracy

#### Phase Detection Accuracy

This metric indicates how accurately the system identifies swing phases. Values closer to 1.0 indicate better accuracy.

### Visualization Elements

In the 3D visualization:
- **Skeleton**: The golfer's body represented as connected joints
- **Ground Grid**: Helps with orientation and perspective
- **Color Coding**: Different body parts are color-coded for clarity
- **Phase Label**: Current swing phase is displayed
- **MPJPE**: Error metric is shown if ground truth data is available

## Troubleshooting

### Common Issues

#### Poor Pose Detection

**Symptoms**: Missing or incorrect landmarks, jittery movement

**Solutions**:
- Ensure good lighting conditions
- Make sure the golfer is fully visible
- Use higher resolution video
- Adjust `detectionCon` parameter in the pose detector

#### Incorrect Swing Phase Detection

**Symptoms**: Phases don't match the actual swing

**Solutions**:
- Train the SVM model with your own labeled data
- Ensure the video captures the entire swing
- Check that the golfer is properly positioned in the frame

#### 3D Visualization Issues

**Symptoms**: Distorted or unrealistic pose

**Solutions**:
- For single camera: try different view angles
- For multi-camera: ensure proper camera calibration
- Adjust smoothing parameters for more natural movement

### Getting Help

If you encounter issues not covered in this guide:
1. Check the documentation in the `docs` directory
2. Look for similar issues in the project repository
3. Contact the development team for support

## Advanced Usage

### Training Custom Models

You can train the swing phase segmenter with your own labeled data:

```python
from main import Pose3DVisualizer

visualizer = Pose3DVisualizer()
visualizer.train_phase_segmenter("data/your_labeled_data.csv")
```

### Extending the System

GolfPose is designed to be modular and extensible. You can:
- Add new evaluation metrics
- Implement custom visualization styles
- Integrate with other golf analysis tools
- Export data for further analysis in other software