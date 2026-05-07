# Multi-View Golf Swing Analysis System
3D Pose Estimation • Swing Phase Segmentation • Rule-Based Feedback

GolfPose is a comprehensive system for analyzing golf swings using computer vision and machine learning techniques. The system provides 3D pose estimation, swing phase detection, and detailed metrics to help golfers improve their technique.

## Features

- **3D Pose Estimation**: Accurate tracking of body movements in three dimensions
- **Multi-Camera Support**: Enhanced accuracy through triangulation of multiple camera views
- **Swing Phase Segmentation**: Automatic detection of swing phases (Address, Takeaway, Backswing, Top, Downswing, Impact, Follow-through)
- **Trajectory Smoothing**: Direct Linear Transformation (DLT) for smooth, natural motion
- **Comprehensive Metrics**: MPJPE (Mean Per Joint Position Error) and other evaluation metrics
- **3D Visualization**: Interactive 3D rendering of the golf swing
- **Deep Learning Integration**: Advanced pose analysis using machine learning models

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorFlow (for deep learning components)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/Asfand6417/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation.git
   cd golfpose
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Single Camera Analysis

For basic golf swing analysis with a single camera:

```python
from main import Pose3DVisualizer

# Initialize the visualizer
visualizer = Pose3DVisualizer()

# Process a video file
visualizer.process_video("path/to/your/video.mp4")

# Create and display the animation
visualizer.animate(show=True, save=True, output_path="output_3d_skeleton.mp4")
```

### Multi-Camera Analysis

For enhanced 3D reconstruction using multiple camera views:

```python
from multicamera.media_pipe_3D_landmark import process_dual_videos_with_single_3d_skeleton

# Process videos from two camera angles
process_dual_videos_with_single_3d_skeleton(
    "path/to/camera1/video.mp4",
    "path/to/camera2/video.mp4",
    output_dir="output_directory"
)
```

## System Components

### Main Module

The main module (`main.py`) provides the core functionality for 3D pose analysis:

- `DirectLinearTransform`: Smooths trajectory data for more natural motion
- `SwingPhaseSegmenter`: Detects golf swing phases using SVM
- `PoseEvaluator`: Calculates metrics like MPJPE for pose accuracy
- `Pose3DVisualizer`: Creates 3D visualizations of the golf swing

### Pose Detection

The `PoseModule.py` provides a simplified interface to MediaPipe's pose detection:

- `poseDetector`: Detects pose landmarks in 2D images
- `findPose`: Processes images to find pose landmarks
- `findPosition`: Extracts landmark coordinates
- `findAngle`: Calculates angles between three points

### Multi-Camera System

The multi-camera components enable more accurate 3D reconstruction:

- `media_pipe_3D_landmark.py`: Processes dual videos to create a single 3D skeleton
- `3D_Reconstruction_through_Triangulation.py`: Implements triangulation for 3D point reconstruction
- `camera_calibration.py`: Tools for calibrating multiple cameras

### Deep Learning Models

The system includes deep learning models for advanced analysis:

- Swing phase classification
- Pose correction suggestions
- Feature importance analysis

## Examples

### Analyzing a Golf Swing Video

```python
from main import Pose3DVisualizer

visualizer = Pose3DVisualizer()
visualizer.process_video("PoseVideos/golf_swing.mp4")
visualizer.animate(save=True, output_path="results/analyzed_swing.mp4")

# Generate a report
report = visualizer.evaluator.generate_report()
print(f"MPJPE: {report['mpjpe']['mean']:.4f}")
print(f"Phase Detection Accuracy: {report['phase_detection']['accuracy']:.2f}")
```

## Documentation

Comprehensive documentation is available in the `docs` directory:

- [Documentation Index](docs/index.md) - Overview of all documentation
- [User Guide](docs/user_guide.md) - Step-by-step instructions for using GolfPose
- [Main Module Documentation](docs/main_module.md) - Details about the core functionality
- [Pose Detection Documentation](docs/pose_detection.md) - Information about the pose detection module
- [Multi-Camera System Documentation](docs/multicamera_system.md) - Documentation for the multi-camera components

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


🧩 Core Technologies

OpenPose – 2D keypoint detection

DLT (Direct Linear Transformation) – 3D pose reconstruction

SVM / MLP – Swing phase classification

Rule-based engine – Posture correction feedback

OpenCV, NumPy, Scikit-learn, Matplotlib, Pandas

📚 Research Context

This project was developed as part of a Master’s thesis focused on golf swing biomechanics analysis using computer vision. It aims to support coaches and athletes in visualizing motion accuracy and improving swing performance through automated analysis.

🧾 Citation

If you use this project in your research or academic work, please cite:

Asfand Yar, "Multi-View Golf Swing Analysis using 3D Pose Estimation and Rule-Based Feedback", 2025.

💬 Contact

Author: Asfand Yar
Role: Web Designer • Computer Vision Expert • Researcher