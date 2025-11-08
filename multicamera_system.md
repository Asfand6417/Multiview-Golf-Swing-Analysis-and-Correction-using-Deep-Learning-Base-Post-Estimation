# Multi-Camera System Documentation

The multi-camera system is a key component of GolfPose that enables more accurate 3D pose reconstruction by combining data from multiple camera views. This documentation covers the main components and functionality of the multi-camera system.

## Overview

The multi-camera system uses triangulation and other techniques to create a more accurate 3D representation of a golf swing. By capturing the same motion from different angles, the system can resolve ambiguities and occlusions that might occur in a single-camera setup.

## Key Components

### 1. Dual Camera 3D Reconstruction (`media_pipe_3D_landmark.py`)

The `media_pipe_3D_landmark.py` file contains the core functionality for processing dual videos and creating a single 3D skeleton visualization.

#### Main Function

```python
process_dual_videos_with_single_3d_skeleton(video1_path, video2_path, output_dir='single_3d_skeleton', fps=30)
```

This function:
- Processes two video files with MediaPipe Pose
- Combines pose data from both cameras based on confidence
- Creates a 3D skeleton visualization
- Generates output videos showing both the original annotated videos and the 3D reconstruction

#### Helper Functions

- `combine_landmarks(landmarks1, landmarks2, confidence1, confidence2)`: Combines landmarks from two camera views based on detection confidence
- `project_3d_to_2d(points_3d, rotation_x, rotation_y, rotation_z, scale, center_x, center_y)`: Projects 3D points to 2D with rotation, scaling, and translation
- `adjust_skeleton_to_ground(landmarks)`: Adjusts the skeleton so feet touch the ground
- `draw_ground_grid(image, points_2d, center_x, center_y, scale, grid_size, grid_step)`: Draws a ground grid for better orientation
- `draw_3d_skeleton(landmarks, width, height, rotation_x, rotation_y, rotation_z)`: Draws a 3D skeleton on a 2D image

### 2. 3D Reconstruction through Triangulation

The `3D_Reconstruction_through_Triangulation.py` file implements triangulation for 3D point reconstruction from multiple 2D views.

#### Key Functions

- Triangulation of 2D points from calibrated cameras
- Optimization of 3D point positions
- Error estimation for reconstructed points

### 3. Camera Calibration

The `camera_calibration.py` file provides tools for calibrating multiple cameras to ensure accurate 3D reconstruction.

#### Calibration Process

1. Capture images of a calibration pattern (e.g., checkerboard) from multiple cameras
2. Detect pattern corners in all images
3. Calculate intrinsic parameters for each camera (focal length, principal point, distortion)
4. Calculate extrinsic parameters (rotation, translation) between cameras
5. Save calibration data for use in 3D reconstruction

#### Key Functions

- `calibrate_cameras(calibration_images_dir)`: Calibrates cameras using images of a calibration pattern
- `save_calibration_data(calibration_data, output_file)`: Saves calibration data to a file
- `load_calibration_data(input_file)`: Loads calibration data from a file

## Directory Structure

The multi-camera system is organized into several directories:

- `3d_reconstruction/`: Contains code for 3D reconstruction algorithms
- `dual_camera_2d_views/`: Handles processing of 2D views from dual cameras
- `dual_camera_3d/`: Basic 3D reconstruction from dual cameras
- `dual_camera_3d_improved/`: Improved 3D reconstruction with additional optimizations
- `single_3d_skeleton/`: Output directory for combined 3D skeleton visualizations

## Usage Examples

### Basic Dual Camera Processing

```python
from multicamera.media_pipe_3D_landmark import process_dual_videos_with_single_3d_skeleton

# Process videos from two camera angles
process_dual_videos_with_single_3d_skeleton(
    "PoseVideos/camera1_swing.mp4",
    "PoseVideos/camera2_swing.mp4",
    output_dir="output/dual_camera_results"
)
```

### Camera Calibration

```python
from multicamera.camera_calibration import calibrate_cameras, save_calibration_data

# Calibrate cameras using images in the calibration directory
calibration_data = calibrate_cameras("calibration_images/")

# Save calibration data for future use
save_calibration_data(calibration_data, "calibration_data.npy")
```

### 3D Reconstruction with Calibrated Cameras

```python
from multicamera.3D_Reconstruction_through_Triangulation import reconstruct_3d_points
from multicamera.camera_calibration import load_calibration_data

# Load calibration data
calibration_data = load_calibration_data("calibration_data.npy")

# Reconstruct 3D points from 2D points in multiple views
points_3d = reconstruct_3d_points(points_2d_view1, points_2d_view2, calibration_data)
```

## Best Practices

For optimal results with the multi-camera system:

1. **Camera Placement**:
   - Position cameras at approximately 90-degree angles to each other
   - Ensure both cameras have a clear view of the subject
   - Maintain consistent lighting across camera views

2. **Calibration**:
   - Calibrate cameras before each session for best accuracy
   - Use a large calibration pattern visible in both camera views
   - Capture multiple calibration images from different positions

3. **Processing**:
   - Ensure videos are synchronized (start recording at the same time)
   - Use videos with the same frame rate
   - Process videos at their native resolution for best accuracy

4. **Visualization**:
   - Adjust rotation angles in the 3D visualization for optimal viewing
   - Use the ground grid for better orientation
   - Compare the 3D reconstruction with the original videos to verify accuracy