"""
Golf Swing Analysis Pipeline

This script ties together all components of the golf swing analysis pipeline:
1. Dual-camera synchronisation
2. 2-D key-point detection with OpenPose
3. DLT triangulation for 3-D reconstruction
4. Swing-phase segmentation with SVM/MLP
5. Rule-based error detection
6. MPJPE evaluation & annotated visual output

Research Objectives:
    1. Accuracy of 3D reconstruction [[45], [46]]
    2. Effectiveness of ML swing-phase segmentation [[44], [47]]
    3. Reliability of rule-based error detection [[48], [49]]
    4. Comparison with single-view & marker-based baselines [[50]-[52]]
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# USAGE INSTRUCTIONS:
# This script uses hardcoded values instead of command-line arguments for easier use.
# To run the script, simply execute: python run_pipeline.py
# To customize the input videos, calibration file, and other options,
# modify the values in the parse_arguments() function below.

# Import our modules
from reconstruction import triangulate_points, triangulate_sequence, align_ground_plane
from visualize_3d import Skeleton3DVisualizer, visualize_sequence
from segmentation import SwingPhaseSegmenter
from error_detection import GolfSwingErrorDetector, detect_errors_in_sequence
from evaluate import PoseEvaluator, evaluate_reconstruction, compare_methods


# Hyperparameters
FPS = 30  # Default frames per second
OUTPUT_DIR = "output"  # Default output directory
MODEL_PATH = "models/swing_phase_segmenter.pkl"  # Default model path


def parse_arguments():
    """Return hardcoded arguments instead of parsing command line."""
    args = argparse.Namespace()

    # Input videos - MODIFY THESE PATHS TO YOUR VIDEO FILES
    args.video1 = "data/videos/front_view.mp4"
    args.video2 = "data/videos/side_view.mp4"

    # Calibration data - MODIFY THIS PATH TO YOUR CALIBRATION FILE
    args.calibration = "data/calibration/camera_calibration.npz"

    # Optional ground truth for evaluation
    args.ground_truth = None  # Set to path if you have ground truth data

    # Output options
    args.output_dir = OUTPUT_DIR
    args.save_video = True  # Set to True to save output videos
    args.show_visualization = True  # Set to True to show 3D visualization

    # Pipeline options
    args.skip_segmentation = False  # Set to True to skip swing phase segmentation
    args.skip_error_detection = False  # Set to True to skip error detection
    args.skip_evaluation = False  # Set to True to skip evaluation

    # Model options
    args.model_path = MODEL_PATH
    args.model_type = "SVM"  # Options: "SVM" or "MLP"

    # Visualization options
    args.bg_color = "white"
    args.fps = FPS

    return args


def load_calibration_data(calibration_path):
    """
    Load camera calibration data from file.

    Args:
        calibration_path: Path to calibration data file

    Returns:
        Dictionary with calibration parameters
    """
    print(f"Loading calibration data from {calibration_path}")

    # Check file extension
    if calibration_path.endswith('.npz'):
        # Load from NumPy file
        data = np.load(calibration_path)
        calibration_data = {
            'camera1_matrix': data['camera1_matrix'],
            'camera2_matrix': data['camera2_matrix'],
            'R': data['R'],
            'T': data['T']
        }
    else:
        # Assume it's a JSON file
        import json
        with open(calibration_path, 'r') as f:
            data = json.load(f)

        # Convert lists to numpy arrays
        calibration_data = {
            'camera1_matrix': np.array(data['camera1_matrix']),
            'camera2_matrix': np.array(data['camera2_matrix']),
            'R': np.array(data['R']),
            'T': np.array(data['T'])
        }

    return calibration_data


def extract_keypoints(video_path, use_mediapipe=True):
    """
    Extract 2D keypoints from a video using MediaPipe or OpenPose.

    Args:
        video_path: Path to video file
        use_mediapipe: Whether to use MediaPipe (True) or OpenPose (False)

    Returns:
        List of keypoint arrays, one per frame
    """
    print(f"Extracting keypoints from {video_path}")

    if use_mediapipe:
        # Use MediaPipe for keypoint extraction
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        keypoints = []
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = pose.process(image_rgb)

            # Extract keypoints if detected
            if results.pose_landmarks:
                frame_keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    h, w, _ = image.shape
                    x = landmark.x * w
                    y = landmark.y * h
                    frame_keypoints.append([x, y])
                keypoints.append(np.array(frame_keypoints))
            else:
                # If no landmarks detected, use the previous frame's landmarks or empty array
                if keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    # Create empty keypoints with the same structure (33 landmarks with x,y=0)
                    keypoints.append(np.zeros((33, 2)))

        # Release resources
        cap.release()
    else:
        # Use OpenPose for keypoint extraction
        # This would require OpenPose to be installed and available
        # For now, we'll just raise an error
        raise NotImplementedError("OpenPose support not implemented yet")

    print(f"Extracted keypoints from {len(keypoints)} frames")
    return keypoints


def run_pipeline(args):
    """
    Run the complete golf swing analysis pipeline.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with pipeline results
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load calibration data
    calibration_data = load_calibration_data(args.calibration)

    # Step 2: Extract 2D keypoints from both videos
    keypoints1 = extract_keypoints(args.video1)
    keypoints2 = extract_keypoints(args.video2)

    # Ensure both keypoint sequences have the same length
    min_length = min(len(keypoints1), len(keypoints2))
    keypoints1 = keypoints1[:min_length]
    keypoints2 = keypoints2[:min_length]

    # Step 3: Triangulate 3D points
    print("Triangulating 3D points...")
    landmarks_3d = triangulate_sequence(keypoints1, keypoints2, calibration_data)

    # Step 4: Swing phase segmentation
    if not args.skip_segmentation:
        print("Performing swing phase segmentation...")
        segmenter = SwingPhaseSegmenter(model_path=args.model_path, model_type=args.model_type)
        phases = segmenter.predict_sequence(landmarks_3d)
    else:
        phases = None

    # Step 5: Error detection
    if not args.skip_error_detection:
        print("Detecting swing errors...")
        error_report = detect_errors_in_sequence(landmarks_3d, segmenter if not args.skip_segmentation else None)

        # Print detected errors
        if error_report['errors']:
            print("\nDetected Errors:")
            for error, description in error_report['errors'].items():
                confidence = error_report['confidence'].get(error, 0)
                print(f"  {description} (Confidence: {confidence:.2f})")
        else:
            print("No errors detected.")
    else:
        error_report = None

    # Step 6: Visualization
    print("Creating 3D visualization...")
    output_video_path = os.path.join(args.output_dir, "3d_visualization.mp4") if args.save_video else None
    animation = visualize_sequence(
        landmarks_3d,
        output_path=output_video_path,
        show=args.show_visualization,
        bg_color=args.bg_color
    )

    # Step 7: Evaluation
    if not args.skip_evaluation and args.ground_truth:
        print("Evaluating reconstruction accuracy...")
        # Load ground truth data
        if args.ground_truth.endswith('.npz'):
            gt_data = np.load(args.ground_truth)
            gt_landmarks = gt_data['landmarks']
        else:
            # Assume it's a JSON file
            import json
            with open(args.ground_truth, 'r') as f:
                gt_data = json.load(f)
                gt_landmarks = np.array(gt_data['landmarks'])

        # Ensure ground truth has the same length as our reconstruction
        min_length = min(len(landmarks_3d), len(gt_landmarks))
        landmarks_3d_eval = landmarks_3d[:min_length]
        gt_landmarks_eval = gt_landmarks[:min_length]

        # Evaluate reconstruction
        evaluation_report = evaluate_reconstruction(landmarks_3d_eval, gt_landmarks_eval)

        # Save evaluation report
        report_path = os.path.join(args.output_dir, "evaluation_report.json")
        import json
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)

        print(f"Evaluation report saved to {report_path}")
    else:
        evaluation_report = None

    # Return results
    return {
        'landmarks_3d': landmarks_3d,
        'phases': phases,
        'error_report': error_report,
        'evaluation_report': evaluation_report
    }


def main():
    """Main function to run the pipeline."""
    # Get hardcoded arguments
    args = parse_arguments()

    try:
        # Run the pipeline
        results = run_pipeline(args)

        print("\nPipeline completed successfully!")
        print(f"Results saved to {args.output_dir}")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
