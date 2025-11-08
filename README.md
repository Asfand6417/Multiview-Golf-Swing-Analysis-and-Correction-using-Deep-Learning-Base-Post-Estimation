🏌️‍♂️ Multi-View Golf Swing Analysis System
3D Pose Estimation • Swing Phase Segmentation • Rule-Based Feedback

This project is a complete computer vision pipeline designed to analyze golf swings using multi-view synchronized cameras. It reconstructs 3D human poses, segments the golf swing into biomechanical phases, evaluates pose accuracy using MPJPE, and provides rule-based visual feedback to help improve swing performance.

🔍 Overview

The system processes dual-camera video input (side and back views) to detect 2D keypoints using OpenPose, reconstruct 3D joint coordinates via DLT (Direct Linear Transformation), and analyze the motion through SVM-based swing phase classification and rule-based evaluation.

Pipeline Summary:

Input: Two synchronized videos (Back View & Side View).

2D Pose Detection: Keypoint extraction via Media Pipe.

3D Pose Reconstruction: DLT triangulation from multi-view data.

Swing Phase Segmentation: Classification using SVM/MLP.

Pose Correction: Rule-based biomechanical evaluation.

Output: Annotated frames and videos with MPJPE, phase label, and feedback.