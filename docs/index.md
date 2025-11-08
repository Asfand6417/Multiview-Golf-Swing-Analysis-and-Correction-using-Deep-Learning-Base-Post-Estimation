# GolfPose Documentation

Welcome to the GolfPose documentation. This documentation provides comprehensive information about the GolfPose system, a tool for analyzing golf swings using computer vision and machine learning techniques.

## Documentation Contents

### Overview

- [README](../README.md) - Project overview, features, installation, and basic usage

### User Documentation

- [User Guide](user_guide.md) - Step-by-step instructions for using GolfPose
  - Installation
  - Quick Start
  - Single Camera Analysis
  - Multi-Camera Analysis
  - Interpreting Results
  - Troubleshooting

### Technical Documentation

- [Main Module](main_module.md) - Documentation for the core functionality
  - DirectLinearTransform
  - SwingPhaseSegmenter
  - PoseEvaluator
  - Pose3DVisualizer

- [Pose Detection](pose_detection.md) - Documentation for the pose detection module
  - poseDetector class
  - MediaPipe integration
  - Usage examples

- [Multi-Camera System](multicamera_system.md) - Documentation for the multi-camera components
  - Dual Camera 3D Reconstruction
  - 3D Reconstruction through Triangulation
  - Camera Calibration

## Getting Started

If you're new to GolfPose, we recommend starting with the [README](../README.md) for an overview of the system, followed by the [User Guide](user_guide.md) for step-by-step instructions on how to use it.

For developers looking to understand or extend the system, the technical documentation provides detailed information about each component.

## Examples

The documentation includes code examples for common tasks:

- Basic pose detection
- 3D visualization of golf swings
- Multi-camera analysis
- Training custom models

## Support

If you encounter any issues or have questions about GolfPose, please:

1. Check the [User Guide](user_guide.md) and [Troubleshooting](user_guide.md#troubleshooting) section
2. Look for similar issues in the project repository
3. Contact the development team for support