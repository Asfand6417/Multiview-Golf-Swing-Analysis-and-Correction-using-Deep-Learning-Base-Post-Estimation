"""
Rule-Based Golf Swing Error Detection

This module implements rule-based logic for detecting common errors in golf swings
using 3D pose data and swing phase information.

Research Objectives:
    3. Reliability of rule-based error detection [[48], [49]]
    
Technical Constraints:
    - Encapsulate hyper-params at top of file for quick tuning
    - Include reliability metrics
"""

import os
import numpy as np
import pandas as pd
from segmentation import SwingPhaseSegmenter


# Hyperparameters for error detection
# Angle thresholds (in degrees)
SPINE_ANGLE_THRESHOLD = 45.0  # Minimum spine angle at address
WRIST_HINGE_THRESHOLD = 80.0  # Minimum wrist hinge at top of backswing
SHOULDER_TURN_THRESHOLD = 80.0  # Minimum shoulder turn in backswing
HIP_TURN_THRESHOLD = 45.0  # Minimum hip turn in backswing
WEIGHT_SHIFT_THRESHOLD = 0.3  # Minimum weight shift (as proportion of body width)
SWING_PLANE_DEVIATION_THRESHOLD = 15.0  # Maximum deviation from ideal swing plane
HEAD_MOVEMENT_THRESHOLD = 0.2  # Maximum head movement (as proportion of body height)

# Confidence thresholds
DETECTION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for reporting an error


class GolfSwingErrorDetector:
    """
    Rule-based detector for common golf swing errors.
    
    Research Alignment:
        Supports Objective 3: Reliability of rule-based error detection [[48], [49]]
        by implementing and evaluating rule-based error detection for golf swings.
    """
    
    def __init__(self, segmenter=None):
        """
        Initialize the error detector.
        
        Args:
            segmenter: Optional SwingPhaseSegmenter instance for phase detection
        """
        self.segmenter = segmenter
        
        # Define common golf swing errors and their descriptions
        self.error_types = {
            'spine_angle': 'Insufficient spine angle at address',
            'wrist_hinge': 'Insufficient wrist hinge in backswing',
            'over_the_top': 'Over-the-top swing path',
            'early_extension': 'Early hip extension',
            'sway': 'Excessive lateral movement away from target in backswing',
            'slide': 'Excessive lateral movement toward target in downswing',
            'head_movement': 'Excessive head movement',
            'flat_shoulder_turn': 'Insufficient shoulder turn',
            'poor_weight_shift': 'Inadequate weight shift',
            'loss_of_posture': 'Loss of posture during swing',
            'swing_plane': 'Swing plane deviation'
        }
        
        # Store detection results
        self.detected_errors = {}
        self.detection_confidence = {}
        self.reliability_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
    
    def detect_errors(self, landmarks_sequence, phases=None):
        """
        Detect errors in a golf swing sequence.
        
        Args:
            landmarks_sequence: List of arrays, each containing 3D landmarks for one frame
            phases: Optional list of swing phases for each frame. If None, will use segmenter
                   to predict phases if available
                   
        Returns:
            Dictionary of detected errors with confidence scores
            
        Research Alignment:
            Error detection rules based on biomechanical principles from [[48]].
        """
        # Reset detection results
        self.detected_errors = {}
        self.detection_confidence = {}
        
        # Get swing phases if not provided
        if phases is None and self.segmenter is not None:
            phases = self.segmenter.predict_sequence(landmarks_sequence)
        
        # If we still don't have phases, we can't perform phase-specific checks
        if phases is None:
            print("Warning: No swing phases provided or segmenter available.")
            # We can still perform some general checks
            phases = ["unknown"] * len(landmarks_sequence)
        
        # Find key frames based on phases
        address_frames = [i for i, phase in enumerate(phases) if phase == "Address"]
        backswing_frames = [i for i, phase in enumerate(phases) if phase == "Backswing"]
        top_frames = [i for i, phase in enumerate(phases) if phase == "Top"]
        downswing_frames = [i for i, phase in enumerate(phases) if phase == "Downswing"]
        impact_frames = [i for i, phase in enumerate(phases) if phase == "Impact"]
        
        # If we have the necessary key frames, perform error detection
        if address_frames and top_frames and impact_frames:
            # Get representative frames for each phase
            address_frame = address_frames[len(address_frames)//2]
            top_frame = top_frames[0]  # First frame of top phase
            impact_frame = impact_frames[0]  # First frame of impact phase
            
            # Check for errors
            self._check_spine_angle(landmarks_sequence[address_frame])
            self._check_wrist_hinge(landmarks_sequence[top_frame])
            self._check_shoulder_turn(landmarks_sequence[address_frame], landmarks_sequence[top_frame])
            self._check_hip_turn(landmarks_sequence[address_frame], landmarks_sequence[top_frame])
            self._check_weight_shift(landmarks_sequence[address_frame], landmarks_sequence[impact_frame])
            self._check_head_movement(landmarks_sequence)
            self._check_swing_plane(landmarks_sequence, backswing_frames + downswing_frames)
            self._check_over_the_top(landmarks_sequence, top_frames, downswing_frames)
            self._check_early_extension(landmarks_sequence, downswing_frames)
            
            # Filter errors by confidence threshold
            self.detected_errors = {
                error: desc for error, desc in self.detected_errors.items()
                if self.detection_confidence.get(error, 0) >= DETECTION_CONFIDENCE_THRESHOLD
            }
        else:
            print("Warning: Could not identify all required swing phases.")
        
        return self.detected_errors
    
    def _check_spine_angle(self, landmarks):
        """Check for proper spine angle at address."""
        # Calculate spine angle (angle between vertical and line from hips to shoulders)
        hips = (landmarks[23] + landmarks[24]) / 2  # Midpoint between left and right hip
        shoulders = (landmarks[11] + landmarks[12]) / 2  # Midpoint between shoulders
        
        # Vector from hips to shoulders
        spine_vector = shoulders - hips
        
        # Angle between spine vector and vertical (y-axis)
        vertical = np.array([0, 1, 0])
        angle = self._angle_between_vectors(spine_vector, vertical)
        
        # Check if spine angle is sufficient
        if angle < SPINE_ANGLE_THRESHOLD:
            self.detected_errors['spine_angle'] = self.error_types['spine_angle']
            self.detection_confidence['spine_angle'] = 1.0 - (angle / SPINE_ANGLE_THRESHOLD)
    
    def _check_wrist_hinge(self, landmarks):
        """Check for proper wrist hinge at top of backswing."""
        # For right-handed golfer, check right wrist angle
        r_shoulder = landmarks[12]
        r_elbow = landmarks[14]
        r_wrist = landmarks[16]
        
        # Calculate angle at elbow
        angle = self._calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        # Check if wrist hinge is sufficient
        if angle > 180 - WRIST_HINGE_THRESHOLD:
            self.detected_errors['wrist_hinge'] = self.error_types['wrist_hinge']
            self.detection_confidence['wrist_hinge'] = (angle - (180 - WRIST_HINGE_THRESHOLD)) / WRIST_HINGE_THRESHOLD
    
    def _check_shoulder_turn(self, address_landmarks, top_landmarks):
        """Check for sufficient shoulder turn in backswing."""
        # Calculate shoulder orientation at address
        address_shoulder_vector = top_landmarks[12] - top_landmarks[11]  # Right to left shoulder
        
        # Calculate shoulder orientation at top
        top_shoulder_vector = top_landmarks[12] - top_landmarks[11]
        
        # Project to horizontal plane (x-z)
        address_shoulder_vector_xz = np.array([address_shoulder_vector[0], 0, address_shoulder_vector[2]])
        top_shoulder_vector_xz = np.array([top_shoulder_vector[0], 0, top_shoulder_vector[2]])
        
        # Calculate turn angle
        angle = self._angle_between_vectors(address_shoulder_vector_xz, top_shoulder_vector_xz)
        
        # Check if shoulder turn is sufficient
        if angle < SHOULDER_TURN_THRESHOLD:
            self.detected_errors['flat_shoulder_turn'] = self.error_types['flat_shoulder_turn']
            self.detection_confidence['flat_shoulder_turn'] = 1.0 - (angle / SHOULDER_TURN_THRESHOLD)
    
    def _check_hip_turn(self, address_landmarks, top_landmarks):
        """Check for sufficient hip turn in backswing."""
        # Calculate hip orientation at address
        address_hip_vector = address_landmarks[24] - address_landmarks[23]  # Right to left hip
        
        # Calculate hip orientation at top
        top_hip_vector = top_landmarks[24] - top_landmarks[23]
        
        # Project to horizontal plane (x-z)
        address_hip_vector_xz = np.array([address_hip_vector[0], 0, address_hip_vector[2]])
        top_hip_vector_xz = np.array([top_hip_vector[0], 0, top_hip_vector[2]])
        
        # Calculate turn angle
        angle = self._angle_between_vectors(address_hip_vector_xz, top_hip_vector_xz)
        
        # Check if hip turn is sufficient
        if angle < HIP_TURN_THRESHOLD:
            self.detected_errors['poor_weight_shift'] = self.error_types['poor_weight_shift']
            self.detection_confidence['poor_weight_shift'] = 1.0 - (angle / HIP_TURN_THRESHOLD)
    
    def _check_weight_shift(self, address_landmarks, impact_landmarks):
        """Check for proper weight shift during swing."""
        # Calculate center of mass at address (approximated by hip midpoint)
        address_com_x = (address_landmarks[23][0] + address_landmarks[24][0]) / 2
        
        # Calculate center of mass at impact
        impact_com_x = (impact_landmarks[23][0] + impact_landmarks[24][0]) / 2
        
        # Calculate body width for normalization
        body_width = np.linalg.norm(address_landmarks[24] - address_landmarks[23])
        
        # Calculate normalized weight shift
        weight_shift = (address_com_x - impact_com_x) / body_width
        
        # Check if weight shift is sufficient
        if weight_shift < WEIGHT_SHIFT_THRESHOLD:
            self.detected_errors['poor_weight_shift'] = self.error_types['poor_weight_shift']
            self.detection_confidence['poor_weight_shift'] = 1.0 - (weight_shift / WEIGHT_SHIFT_THRESHOLD)
    
    def _check_head_movement(self, landmarks_sequence):
        """Check for excessive head movement during swing."""
        # Track head position throughout swing
        head_positions = [landmarks[0] for landmarks in landmarks_sequence]
        
        # Calculate maximum head movement
        head_x = [pos[0] for pos in head_positions]
        head_y = [pos[1] for pos in head_positions]
        head_z = [pos[2] for pos in head_positions]
        
        max_head_movement_x = max(head_x) - min(head_x)
        max_head_movement_y = max(head_y) - min(head_y)
        max_head_movement_z = max(head_z) - min(head_z)
        
        # Calculate body height for normalization
        body_height = np.linalg.norm(landmarks_sequence[0][0] - landmarks_sequence[0][24])  # Head to right hip
        
        # Normalize head movement
        normalized_movement = max(max_head_movement_x, max_head_movement_y, max_head_movement_z) / body_height
        
        # Check if head movement is excessive
        if normalized_movement > HEAD_MOVEMENT_THRESHOLD:
            self.detected_errors['head_movement'] = self.error_types['head_movement']
            self.detection_confidence['head_movement'] = (normalized_movement - HEAD_MOVEMENT_THRESHOLD) / HEAD_MOVEMENT_THRESHOLD
    
    def _check_swing_plane(self, landmarks_sequence, swing_frames):
        """Check for swing plane consistency."""
        if not swing_frames:
            return
            
        # Track club path (approximated by right wrist)
        club_path = [landmarks_sequence[i][16] for i in swing_frames]
        
        # Fit a plane to the club path
        if len(club_path) >= 3:
            # Convert to numpy array
            points = np.array(club_path)
            
            # Calculate centroid
            centroid = np.mean(points, axis=0)
            
            # Form the matrix A
            A = points - centroid
            
            # SVD decomposition
            _, _, vh = np.linalg.svd(A)
            
            # Normal vector of the best-fitting plane
            normal = vh[2]
            
            # Calculate deviation from ideal swing plane
            # Ideal swing plane is often around 70 degrees from horizontal
            ideal_normal = np.array([0, np.sin(np.radians(70)), np.cos(np.radians(70))])
            
            # Calculate angle between actual and ideal plane normals
            deviation = np.degrees(np.arccos(np.abs(np.dot(normal, ideal_normal))))
            
            # Check if swing plane deviation is excessive
            if deviation > SWING_PLANE_DEVIATION_THRESHOLD:
                self.detected_errors['swing_plane'] = self.error_types['swing_plane']
                self.detection_confidence['swing_plane'] = (deviation - SWING_PLANE_DEVIATION_THRESHOLD) / SWING_PLANE_DEVIATION_THRESHOLD
    
    def _check_over_the_top(self, landmarks_sequence, top_frames, downswing_frames):
        """Check for over-the-top swing path."""
        if not top_frames or not downswing_frames:
            return
            
        # Get the first few frames of downswing
        early_downswing = downswing_frames[:min(5, len(downswing_frames))]
        
        if not early_downswing:
            return
            
        # Get top of backswing position
        top_position = landmarks_sequence[top_frames[-1]][16]  # Right wrist at top
        
        # Track right wrist path in early downswing
        wrist_path = [landmarks_sequence[i][16] for i in early_downswing]
        
        # Check if path moves left (for right-handed golfer) in early downswing
        # This is a sign of over-the-top move
        if wrist_path[0][0] < top_position[0]:  # X-coordinate decreases (moves left)
            self.detected_errors['over_the_top'] = self.error_types['over_the_top']
            
            # Calculate confidence based on how much left movement
            left_movement = top_position[0] - wrist_path[0][0]
            body_width = np.linalg.norm(landmarks_sequence[0][24] - landmarks_sequence[0][23])  # Hip width
            self.detection_confidence['over_the_top'] = min(1.0, left_movement / (body_width * 0.2))
    
    def _check_early_extension(self, landmarks_sequence, downswing_frames):
        """Check for early hip extension during downswing."""
        if not downswing_frames:
            return
            
        # Get hip positions during downswing
        hip_positions = [(landmarks_sequence[i][23] + landmarks_sequence[i][24]) / 2 for i in downswing_frames]
        
        # Check if hips move toward the ball (z-direction) during downswing
        if len(hip_positions) >= 2:
            start_z = hip_positions[0][2]
            max_z = max(pos[2] for pos in hip_positions)
            
            # Calculate body depth for normalization
            body_depth = np.linalg.norm(landmarks_sequence[0][2] - landmarks_sequence[0][24])  # Nose to right hip
            
            # Normalize movement
            normalized_extension = (max_z - start_z) / body_depth
            
            # Check if extension is excessive
            if normalized_extension > 0.15:  # Threshold for early extension
                self.detected_errors['early_extension'] = self.error_types['early_extension']
                self.detection_confidence['early_extension'] = min(1.0, normalized_extension / 0.3)
    
    def _calculate_angle(self, a, b, c):
        """Calculate angle between three 3D points."""
        ba = a - b
        bc = c - b

        # Calculate dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)
    
    def _angle_between_vectors(self, v1, v2):
        """Calculate the angle between two vectors."""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)

        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    
    def evaluate_reliability(self, detected_errors, ground_truth_errors):
        """
        Evaluate the reliability of error detection against ground truth.
        
        Args:
            detected_errors: Dictionary of detected errors
            ground_truth_errors: Dictionary of ground truth errors
            
        Returns:
            Dictionary with reliability metrics
            
        Research Alignment:
            Reliability metrics follow [[49]] for evaluating error detection performance.
        """
        # Reset reliability metrics
        self.reliability_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        # Check all possible error types
        for error in self.error_types:
            if error in detected_errors and error in ground_truth_errors:
                self.reliability_metrics['true_positives'] += 1
            elif error in detected_errors and error not in ground_truth_errors:
                self.reliability_metrics['false_positives'] += 1
            elif error not in detected_errors and error not in ground_truth_errors:
                self.reliability_metrics['true_negatives'] += 1
            elif error not in detected_errors and error in ground_truth_errors:
                self.reliability_metrics['false_negatives'] += 1
        
        # Calculate derived metrics
        tp = self.reliability_metrics['true_positives']
        fp = self.reliability_metrics['false_positives']
        tn = self.reliability_metrics['true_negatives']
        fn = self.reliability_metrics['false_negatives']
        
        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Add derived metrics
        self.reliability_metrics.update({
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'accuracy': accuracy
        })
        
        return self.reliability_metrics
    
    def get_error_report(self):
        """
        Generate a detailed error report with confidence scores.
        
        Returns:
            Dictionary with error details and confidence scores
        """
        report = {
            'errors': self.detected_errors,
            'confidence': self.detection_confidence,
            'reliability': self.reliability_metrics
        }
        
        return report


def detect_errors_in_sequence(landmarks_sequence, segmenter=None):
    """
    Detect errors in a golf swing sequence.
    
    Args:
        landmarks_sequence: List of arrays, each containing 3D landmarks for one frame
        segmenter: Optional SwingPhaseSegmenter instance for phase detection
        
    Returns:
        Error report dictionary
    """
    detector = GolfSwingErrorDetector(segmenter)
    detector.detect_errors(landmarks_sequence)
    return detector.get_error_report()


if __name__ == "__main__":
    # Example usage
    print("Golf Swing Error Detection module - Run through the main pipeline")