# Multi-View Golf Swing Analysis and Correction Using Deep-Learning-Based Pose Estimation

This project implements a comprehensive pipeline for golf swing analysis using dual-camera views, 3D reconstruction, and machine learning techniques.

## Pipeline Overview

1. **Dual-camera synchronisation**: Process synchronized videos from two camera views
2. **2-D key-point detection**: Extract pose landmarks using MediaPipe
3. **DLT triangulation**: Reconstruct 3D pose from 2D views
4. **Swing-phase segmentation**: Identify swing phases using SVM/MLP models
5. **Rule-based error detection**: Detect common swing errors
6. **MPJPE evaluation**: Evaluate accuracy and provide annotated visual output

## 🔬 Research Alignment

This implementation addresses four key research objectives:

### 1. Accuracy of 3D reconstruction [[45], [46]]
- **reconstruction.py**: Implements DLT triangulation with reprojection error checking
- **visualize_3d.py**: Provides clear visualization of the reconstructed skeleton
- **evaluate.py**: Calculates MPJPE and other accuracy metrics

### 2. Effectiveness of ML swing-phase segmentation [[44], [47]]
- **segmentation.py**: Implements and evaluates SVM and MLP models for phase detection
- Feature extraction based on biomechanical principles from [[44]]
- Evaluation metrics follow [[47]] for comparing segmentation performance

### 3. Reliability of rule-based error detection [[48], [49]]
- **error_detection.py**: Implements rule-based error detection with confidence scores
- Error detection rules based on biomechanical principles from [[48]]
- Reliability metrics follow [[49]] for evaluating error detection performance

### 4. Comparison with single-view & marker-based baselines [[50]-[52]]
- **evaluate.py**: Enables comparison with single-view and marker-based methods
- **visualize_3d.py**: Enables visual comparison of different reconstruction methods
- Comparison methodology follows [[50]-[52]] for evaluating against baselines

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/golf-swing-analysis.git
cd golf-swing-analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

```bash
python run_pipeline.py --video1 path/to/camera1.mp4 --video2 path/to/camera2.mp4 --calibration path/to/calibration.json --save_video --show_visualization
```

### Optional Arguments

```
--output_dir OUTPUT_DIR   Directory to save outputs
--save_video              Save output videos
--show_visualization      Show 3D visualization
--skip_segmentation       Skip swing phase segmentation
--skip_error_detection    Skip error detection
--skip_evaluation         Skip evaluation
--model_path MODEL_PATH   Path to swing phase segmentation model
--model_type {SVM,MLP}    Type of segmentation model
--bg_color BG_COLOR       Background color for visualization
--fps FPS                 Frame rate for output videos
--ground_truth GT_PATH    Path to ground truth 3D data for evaluation
```

### Using Individual Components

#### 3D Reconstruction

```python
from reconstruction import triangulate_sequence, align_ground_plane

# Load calibration data and 2D keypoints
# ...

# Triangulate 3D points
landmarks_3d = triangulate_sequence(keypoints1, keypoints2, calibration_data)

# Align to ground plane
aligned_landmarks = align_ground_plane(landmarks_3d)
```

#### Visualization

```python
from visualize_3d import visualize_sequence

# Visualize 3D landmarks
visualize_sequence(landmarks_3d, output_path="output.mp4", show=True, bg_color="white")
```

#### Swing Phase Segmentation

```python
from segmentation import SwingPhaseSegmenter

# Create segmenter
segmenter = SwingPhaseSegmenter(model_type="SVM")

# Train on labeled data
metrics = segmenter.train(landmark_sequences, labels)

# Predict phases
phases = segmenter.predict_sequence(landmarks_3d)
```

#### Error Detection

```python
from error_detection import detect_errors_in_sequence

# Detect errors
error_report = detect_errors_in_sequence(landmarks_3d, segmenter)

# Print detected errors
for error, description in error_report['errors'].items():
    confidence = error_report['confidence'].get(error, 0)
    print(f"{description} (Confidence: {confidence:.2f})")
```

#### Evaluation

```python
from evaluate import evaluate_reconstruction, compare_methods

# Evaluate reconstruction accuracy
report = evaluate_reconstruction(pred_sequence, gt_sequence)

# Compare with baselines
evaluator = compare_methods(method_results, baseline_results, output_path="comparison.png")
```

## Technical Specifications

- **Language**: Python ≥3.9
- **Libraries**: OpenCV, NumPy, Pandas, scikit-learn, Matplotlib
- **3D axes**: x = left→right, y = up, z = forward (toward ball)
- **Ground plane**: Feet anchored to z=0
- **Visualization**: Single line colour (#00AEEF) for limbs, red key-points (#FF0000)

## Key Features

- **Combined 3D skeleton**: Merges left/right camera views into a single 3D skeleton per frame
- **Transparent background**: Removes black background in 3D output for better visualization
- **Ground plane anchoring**: Ensures feet rest on the z=0 ground plane
- **Consistent visualization**: Uses specified colors for limbs and keypoints
- **Comprehensive evaluation**: Includes MPJPE, reprojection error, and comparison with baselines
- **Detailed error detection**: Identifies common golf swing errors with confidence scores

## Citations

- [44] Reference for swing-phase segmentation biomechanical principles
- [45] Reference for 3D reconstruction accuracy metrics
- [46] Reference for reprojection error evaluation
- [47] Reference for phase segmentation evaluation metrics
- [48] Reference for error detection biomechanical principles
- [49] Reference for error detection reliability metrics
- [50] Reference for baseline comparison methodology
- [51] Reference for single-view baseline comparison
- [52] Reference for marker-based baseline comparison