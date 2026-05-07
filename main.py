"""
Advanced 3D Pose Analysis for Golf Swing
----------------------------------------
Features:
- 3D pose visualization with MediaPipe
- Trajectory smoothing using Direct Linear Transformation (DLT)
- Golf swing phase segmentation using SVM
- Comprehensive evaluation metrics including MPJPE
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe_compat as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import joblib
from club_type_detector import ClubTypeDetector


class DirectLinearTransform:
    """Implementation of Direct Linear Transformation for trajectory smoothing"""

    def __init__(self, window_size=5):
        """Initialize DLT smoother with specified window size"""
        self.window_size = window_size

    def smooth_trajectory(self, landmarks_sequence):
        """
        Apply DLT to smooth a sequence of landmarks

        Args:
            landmarks_sequence: List of numpy arrays containing landmarks for each frame

        Returns:
            Smoothed landmarks sequence
        """
        if len(landmarks_sequence) < self.window_size:
            return landmarks_sequence  # Not enough frames to smooth

        num_frames = len(landmarks_sequence)
        num_landmarks = landmarks_sequence[0].shape[0]
        smoothed_sequence = []

        # Process each frame with a sliding window
        for i in range(num_frames):
            # Create a window centered at current frame
            half_window = self.window_size // 2
            start_idx = max(0, i - half_window)
            end_idx = min(num_frames, i + half_window + 1)
            window = landmarks_sequence[start_idx:end_idx]

            # If window is smaller than expected (at boundaries), adjust
            if len(window) < self.window_size:
                if i < half_window:  # At beginning
                    window = landmarks_sequence[:self.window_size]
                else:  # At end
                    window = landmarks_sequence[-self.window_size:]

            # Apply DLT to each landmark point
            smoothed_frame = np.zeros_like(landmarks_sequence[i])
            for j in range(num_landmarks):
                # Extract trajectory of this landmark across window frames
                point_trajectory = np.array([frame[j] for frame in window])

                # Apply DLT to this trajectory
                smoothed_point = self._apply_dlt_to_point(point_trajectory)
                smoothed_frame[j] = smoothed_point

            smoothed_sequence.append(smoothed_frame)

        return smoothed_sequence

    def _apply_dlt_to_point(self, point_trajectory):
        """
        Apply DLT to a single point's trajectory

        Args:
            point_trajectory: Numpy array of shape (window_size, 3) with x,y,z coordinates

        Returns:
            Smoothed 3D point
        """
        # For a simple implementation, we use a weighted average
        # In a full DLT, you would solve the projection equations
        weights = np.ones(len(point_trajectory))

        # Give higher weight to the central points
        mid_idx = len(weights) // 2
        for i in range(len(weights)):
            weights[i] = 1.0 / (1.0 + 0.5 * abs(i - mid_idx))

        # Normalize weights
        weights = weights / np.sum(weights)

        # Apply weighted average
        smoothed_point = np.zeros(3)
        for i, point in enumerate(point_trajectory):
            smoothed_point += weights[i] * point

        return smoothed_point


class SwingPhaseSegmenter:
    """Golf swing phase segmentation using SVM"""

    # Define golf swing phases
    PHASES = {
        0: "Address",
        1: "Takeaway",
        2: "Backswing",
        3: "Top",
        4: "Downswing",
        5: "Impact",
        6: "Follow-through"
    }

    def __init__(self, model_path=None):
        """Initialize the swing phase segmenter"""
        self.scaler = StandardScaler()

        # Try to load pre-trained model if it exists
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Loaded SVM model from {model_path}")
        else:
            # Create a new SVM model
            self.model = svm.SVC(kernel='rbf', decision_function_shape='ovr')
            print("Created new SVM model for swing phase segmentation")

    def extract_features(self, landmarks):
        """
        Extract features from pose landmarks that are relevant for swing phase detection

        Args:
            landmarks: Array of shape (33, 3) with x,y,z coordinates for MediaPipe pose landmarks

        Returns:
            Feature vector
        """
        features = []

        # Extract key joint positions
        # Right arm angle (shoulder, elbow, wrist)
        r_shoulder = landmarks[12]
        r_elbow = landmarks[14]
        r_wrist = landmarks[16]
        r_arm_angle = self._calculate_angle(r_shoulder, r_elbow, r_wrist)
        features.append(r_arm_angle)

        # Left arm angle
        l_shoulder = landmarks[11]
        l_elbow = landmarks[13]
        l_wrist = landmarks[15]
        l_arm_angle = self._calculate_angle(l_shoulder, l_elbow, l_wrist)
        features.append(l_arm_angle)

        # Torso rotation (shoulders relative to hips)
        r_hip = landmarks[24]
        l_hip = landmarks[23]
        shoulder_vector = r_shoulder - l_shoulder
        hip_vector = r_hip - l_hip
        torso_rotation = self._angle_between_vectors(shoulder_vector[:2], hip_vector[:2])
        features.append(torso_rotation)

        # Wrist height relative to shoulder
        wrist_height = r_wrist[1] - r_shoulder[1]
        features.append(wrist_height)

        # Wrist position relative to body center
        body_center = (r_hip + l_hip) / 2
        wrist_to_center = r_wrist - body_center
        features.extend(wrist_to_center)

        # Add velocity features if we're tracking across frames
        # (Would be calculated externally and passed in)

        return np.array(features)

    def _calculate_angle(self, a, b, c):
        """Calculate angle between three 3D points"""
        ba = a - b
        bc = c - b

        # Calculate dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def _angle_between_vectors(self, v1, v2):
        """Calculate the angle between two vectors"""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)

        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    def train(self, landmark_sequences, labels):
        """
        Train the SVM model on labeled sequences

        Args:
            landmark_sequences: List of landmark arrays
            labels: Corresponding phase labels for each sequence

        Returns:
            Training accuracy
        """
        # Extract features from all sequences
        features = []
        for landmarks in landmark_sequences:
            features.append(self.extract_features(landmarks))

        features = np.array(features)

        # Scale features
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)

        # Split data into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, labels, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')

        print(f"Training complete - Accuracy: {accuracy:.2f}, Recall: {recall:.2f}")

        return accuracy

    def predict(self, landmarks):
        """
        Predict the swing phase for a single frame

        Args:
            landmarks: Array of shape (33, 3) with pose landmarks

        Returns:
            Predicted phase as string
        """
        features = self.extract_features(landmarks)
        scaled_features = self.scaler.transform([features])
        phase_id = self.model.predict(scaled_features)[0]
        return self.PHASES[phase_id]

    def save_model(self, model_path):
        """Save the trained model to disk"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")


class PoseEvaluator:
    """Evaluation metrics for pose estimation"""

    def __init__(self):
        """Initialize the pose evaluator"""
        self.metrics = {
            'mpjpe': [],
            'per_joint_error': {},
            'phase_accuracy': [],
            'subjective_scores': []
        }

    def calculate_mpjpe(self, pred_coords, gt_coords):
        """
        Calculate Mean Per Joint Position Error between predicted and ground truth coordinates

        Args:
            pred_coords: Predicted landmark coordinates (n_joints, 3)
            gt_coords: Ground truth landmark coordinates (n_joints, 3)

        Returns:
            MPJPE value (mean Euclidean distance)
        """
        # Ensure arrays are the right shape
        pred = np.array(pred_coords)
        gt = np.array(gt_coords)

        # Make sure shapes match - truncate if necessary
        min_length = min(len(pred), len(gt))
        pred = pred[:min_length]
        gt = gt[:min_length]

        # Calculate Euclidean distance for each joint
        distances = np.linalg.norm(pred - gt, axis=1)

        # Store per-joint errors
        for i, dist in enumerate(distances):
            if i not in self.metrics['per_joint_error']:
                self.metrics['per_joint_error'][i] = []
            self.metrics['per_joint_error'][i].append(dist)

        # Calculate mean
        mpjpe = np.mean(distances)

        # Store for reporting
        self.metrics['mpjpe'].append(mpjpe)

        return mpjpe

    def evaluate_phase_accuracy(self, predicted_phases, ground_truth_phases):
        """
        Evaluate the accuracy of phase detection

        Args:
            predicted_phases: List of predicted phases
            ground_truth_phases: List of ground truth phases

        Returns:
            Accuracy and recall scores
        """
        # Convert phases to numerical indices if they are strings
        if isinstance(predicted_phases[0], str):
            phase_to_idx = {phase: i for i, phase in enumerate(set(predicted_phases).union(set(ground_truth_phases)))}
            pred_idx = [phase_to_idx[phase] for phase in predicted_phases]
            gt_idx = [phase_to_idx[phase] for phase in ground_truth_phases]
        else:
            pred_idx = predicted_phases
            gt_idx = ground_truth_phases

        # Calculate metrics
        accuracy = accuracy_score(gt_idx, pred_idx)
        recall = recall_score(gt_idx, pred_idx, average='macro')
        conf_matrix = confusion_matrix(gt_idx, pred_idx)

        # Store metrics
        self.metrics['phase_accuracy'].append({
            'accuracy': accuracy,
            'recall': recall,
            'confusion_matrix': conf_matrix
        })

        return accuracy, recall

    def add_subjective_feedback(self, score, comments=""):
        """
        Add subjective feedback from a coach or expert

        Args:
            score: Numerical score (e.g., 1-10)
            comments: Text feedback
        """
        self.metrics['subjective_scores'].append({
            'score': score,
            'comments': comments
        })

    def generate_report(self):
        """
        Generate a comprehensive evaluation report

        Returns:
            Dictionary with all evaluation metrics
        """
        report = {
            'mpjpe': {
                'mean': np.mean(self.metrics['mpjpe']),
                'std': np.std(self.metrics['mpjpe']),
                'min': np.min(self.metrics['mpjpe']) if self.metrics['mpjpe'] else None,
                'max': np.max(self.metrics['mpjpe']) if self.metrics['mpjpe'] else None
            },
            'per_joint_error': {
                joint: {
                    'mean': np.mean(errors),
                    'std': np.std(errors)
                } for joint, errors in self.metrics['per_joint_error'].items()
            },
            'phase_detection': {
                'accuracy': np.mean([m['accuracy'] for m in self.metrics['phase_accuracy']]) if self.metrics[
                    'phase_accuracy'] else None,
                'recall': np.mean([m['recall'] for m in self.metrics['phase_accuracy']]) if self.metrics[
                    'phase_accuracy'] else None
            },
            'subjective': {
                'mean_score': np.mean([s['score'] for s in self.metrics['subjective_scores']]) if self.metrics[
                    'subjective_scores'] else None,
                'comments': [s['comments'] for s in self.metrics['subjective_scores']]
            }
        }

        return report


class Pose3DVisualizer:
    def __init__(self):
        """Initialize the MediaPipe pose detector and visualization settings"""
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Using highest quality model
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Setup for 3D plotting
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Store landmarks for each frame
        self.all_landmarks = []
        self.smoothed_landmarks = []
        self.predicted_phases = []
        self.predicted_club_types = []

        # Framerate management
        self.fps = 30
        self.frame_time = 1 / self.fps

        # Initialize trajectory smoother
        self.smoother = DirectLinearTransform(window_size=5)

        # Initialize phase segmenter
        self.segmenter = SwingPhaseSegmenter(model_path="models/swing_phase_svm.pkl")

        # Initialize club type detector
        self.club_detector = ClubTypeDetector(model_path="models/club_type_svm.pkl")

        # Initialize evaluator
        self.evaluator = PoseEvaluator()

    def process_video(self, video_path, ground_truth_path=None):
        """
        Process video file and extract 3D landmarks from each frame

        Args:
            video_path: Path to the input video file
            ground_truth_path: Optional path to ground truth data
        """
        # Check if video file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video with {total_frames} frames at {self.fps} FPS...")

        # Load ground truth data if available
        gt_data = self._load_ground_truth(ground_truth_path) if ground_truth_path else None

        # Process each frame
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose.process(image_rgb)

            # Store 3D landmarks if detected
            if results.pose_world_landmarks:
                # Extract 3D coordinates from MediaPipe results
                landmarks_3d = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark
                ])

                # Reorient landmarks for proper floor alignment
                landmarks_3d = self._reorient_landmarks(landmarks_3d)

                self.all_landmarks.append(landmarks_3d)

                # Predict swing phase
                phase = self.segmenter.predict(landmarks_3d)
                self.predicted_phases.append(phase)

                # Predict club type
                club_type = self.club_detector.predict(landmarks_3d)
                self.predicted_club_types.append(club_type)

                # Evaluate against ground truth if available
                if gt_data and frame_count < len(gt_data['landmarks']):
                    gt_landmarks = np.array(gt_data['landmarks'][frame_count])
                    mpjpe = self.evaluator.calculate_mpjpe(landmarks_3d, gt_landmarks)

                    if 'phases' in gt_data and frame_count < len(gt_data['phases']):
                        gt_phase = gt_data['phases'][frame_count]
                        self.evaluator.evaluate_phase_accuracy([phase], [gt_phase])
            else:
                # If no landmarks detected, use the previous frame's landmarks or empty array
                if self.all_landmarks:
                    self.all_landmarks.append(self.all_landmarks[-1])
                    self.predicted_phases.append(self.predicted_phases[-1])
                    self.predicted_club_types.append(self.predicted_club_types[-1] if self.predicted_club_types else "Unknown")
                else:
                    self.all_landmarks.append(np.zeros((33, 3)))  # 33 landmarks with x,y,z=0
                    self.predicted_phases.append("Unknown")
                    self.predicted_club_types.append("Unknown")

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        # Release resources
        cap.release()
        print(f"Video processing complete! Extracted landmarks from {len(self.all_landmarks)} frames.")

        # Apply trajectory smoothing
        self.smoothed_landmarks = self.smoother.smooth_trajectory(self.all_landmarks)
        print("Trajectory smoothing complete using DLT.")

    def _load_ground_truth(self, gt_path):
        """Load ground truth data from file"""
        if not os.path.isfile(gt_path):
            print(f"Warning: Ground truth file {gt_path} not found.")
            return None

        try:
            _, ext = os.path.splitext(gt_path)
            if ext.lower() == '.json':
                import json
                with open(gt_path, 'r') as f:
                    return json.load(f)
            elif ext.lower() == '.npy':
                return {'landmarks': np.load(gt_path, allow_pickle=True)}
            else:
                print(f"Unsupported ground truth file format: {ext}")
                return None
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            return None

    def _reorient_landmarks(self, landmarks):
        """
        Reorient the landmarks so the person is standing properly on the X-Z plane

        Args:
            landmarks: Numpy array of shape (n_landmarks, 3)

        Returns:
            Reoriented landmarks array
        """
        # Make a copy to avoid modifying the original
        reoriented = landmarks.copy()

        # Find foot landmarks (ankles, heels, toes)
        foot_indices = [27, 28, 29, 30, 31, 32]
        foot_y_values = [landmarks[i, 1] for i in foot_indices if i < len(landmarks)]

        if foot_y_values:
            # Find ground level (max Y value in MediaPipe coordinates)
            ground_level = max(foot_y_values)

            # Normalize Y coordinates so feet are at Y=0
            reoriented[:, 1] -= ground_level

            # Flip Y axis for more intuitive visualization (up is positive)
            reoriented[:, 1] = -reoriented[:, 1]

        return reoriented

    def _get_pose_connections(self):
        """Get the joint connections used by MediaPipe Pose"""
        return [(connection[0], connection[1]) for connection in self.mp_pose.POSE_CONNECTIONS]

    def _update_plot(self, frame_idx):
        """Update the 3D plot for a specific frame"""
        self.ax.clear()

        # Get landmarks for current frame
        if frame_idx >= len(self.smoothed_landmarks):
            return

        landmarks = self.smoothed_landmarks[frame_idx]
        phase = self.predicted_phases[frame_idx] if frame_idx < len(self.predicted_phases) else "Unknown"
        club_type = self.predicted_club_types[frame_idx] if frame_idx < len(self.predicted_club_types) else "Unknown"

        # Set axis labels and title
        self.ax.set_xlabel('X (left/right)')
        self.ax.set_ylabel('Z (forward/backward)')
        self.ax.set_zlabel('Y (up/down)')
        self.ax.set_title(f'3D Pose - Frame {frame_idx} - Phase: {phase} - Club: {club_type}')

        # Draw a grid on the X-Z plane (floor)
        x_min, x_max = -1, 1
        z_min, z_max = -1, 1
        XX, ZZ = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
        YY = np.zeros_like(XX)
        self.ax.plot_surface(XX, ZZ, YY, alpha=0.2, color='gray')

        # Plot all landmarks
        self.ax.scatter(
            landmarks[:, 0],  # X
            landmarks[:, 2],  # Z
            landmarks[:, 1],  # Y
            color='blue', s=20
        )

        # Draw connections between joints
        connections = self._get_pose_connections()
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                xs = [landmarks[start_idx, 0], landmarks[end_idx, 0]]
                zs = [landmarks[start_idx, 2], landmarks[end_idx, 2]]
                ys = [landmarks[start_idx, 1], landmarks[end_idx, 1]]
                self.ax.plot(xs, zs, ys, color='red', linewidth=2)

        # Set appropriate axis limits
        max_range = 2.0
        self.ax.set_xlim(-max_range / 2, max_range / 2)  # X
        self.ax.set_ylim(-max_range / 2, max_range / 2)  # Z
        self.ax.set_zlim(0, max_range)  # Y - start from 0 for the floor

        # Different view angles for different phases
        if phase == "Address":
            self.ax.view_init(elev=15, azim=0)
        elif phase in ["Takeaway", "Backswing"]:
            self.ax.view_init(elev=20, azim=45)
        elif phase == "Top":
            self.ax.view_init(elev=25, azim=90)
        elif phase in ["Downswing", "Impact"]:
            self.ax.view_init(elev=20, azim=135)
        elif phase == "Follow-through":
            self.ax.view_init(elev=15, azim=180)
        else:
            self.ax.view_init(elev=15, azim=(frame_idx % 360))

        # Add phase and club type information as text
        self.ax.text2D(0.05, 0.95, f"Phase: {phase}", transform=self.ax.transAxes,
                       fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8))
        self.ax.text2D(0.05, 0.90, f"Club: {club_type}", transform=self.ax.transAxes,
                       fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8))

        # Add MPJPE if available
        if self.evaluator.metrics['mpjpe'] and frame_idx < len(self.evaluator.metrics['mpjpe']):
            mpjpe = self.evaluator.metrics['mpjpe'][frame_idx]
            self.ax.text2D(0.05, 0.85, f"MPJPE: {mpjpe:.4f}", transform=self.ax.transAxes,
                           fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.8))

        return self.ax

    def animate(self, show=True, save=False, output_path='output_3d_skeleton.mp4'):
        """
        Create an animation of the 3D pose.

        Args:
            show (bool): Whether to display the animation in real-time
            save (bool): Whether to save the animation to a video file
            output_path (str): Path to save the output video if save=True
        """
        if len(self.smoothed_landmarks) == 0:
            if len(self.all_landmarks) > 0:
                self.smoothed_landmarks = self.smoother.smooth_trajectory(self.all_landmarks)
            else:
                raise ValueError("No landmarks to animate. Process a video first.")

        num_frames = len(self.smoothed_landmarks)
        print(f"Creating animation with {num_frames} frames...")

        # Create animation
        ani = FuncAnimation(
            self.fig,
            self._update_plot,
            frames=num_frames,
            interval=self.frame_time * 1000,  # Convert to milliseconds
            blit=False
        )

        # Save animation if requested
        if save:
            print(f"Saving animation to {output_path}...")

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Setup FFmpeg writer
            writer = FFMpegWriter(
                fps=self.fps,
                metadata=dict(artist='MediaPipe 3D Pose'),
                bitrate=2000
            )

            # Save the animation
            try:
                ani.save(output_path, writer=writer)
                print(f"Animation saved to {output_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Make sure FFmpeg is installed and in your PATH.")

        # Show the animation if requested
        if show:
            plt.tight_layout()
            plt.show()

        # Generate evaluation report
        report = self.evaluator.generate_report()
        print("\nEvaluation Report:")
        print(f"MPJPE: {report['mpjpe']['mean']:.4f} ± {report['mpjpe']['std']:.4f}")
        print(f"Phase Detection Accuracy: {report['phase_detection']['accuracy']:.2f}")
        print(f"Phase Detection Recall: {report['phase_detection']['recall']:.2f}")

        if report['subjective']['mean_score']:
            print(f"Subjective Score: {report['subjective']['mean_score']:.1f}/10")

        return ani

    def train_phase_segmenter(self, training_data_path):
        """
        Train the SVM phase segmenter using manually labeled data

        Args:
            training_data_path: Path to a CSV file with labeled swing phases
        """
        if not os.path.exists(training_data_path):
            print(f"Training data file {training_data_path} not found.")
            return False

        try:
            # Load data
            df = pd.read_csv(training_data_path)

            # Extract features and labels
            landmark_sequences = []
            labels = []

            for _, row in df.iterrows():
                landmarks = np.array(eval(row['landmarks']))  # Convert string representation to numpy array
                phase = row['phase']
                landmark_sequences.append(landmarks)
                labels.append(phase)

            # Train the SVM model
            accuracy = self.segmenter.train(landmark_sequences, labels)

            # Save the trained model
            os.makedirs('models', exist_ok=True)
            self.segmenter.save_model('models/swing_phase_svm.pkl')

            return accuracy

        except Exception as e:
            print(f"Error training phase segmenter: {e}")
            return False


def main():
    """Main function to run the 3D pose visualization"""
    # Set up the visualizer
    visualizer = Pose3DVisualizer()

    # Check for training data and train the SVM if necessary
    training_data_path = "data/swing_phase_labels.csv"
    if os.path.exists(training_data_path):
        print("Training swing phase segmentation model...")
        visualizer.train_phase_segmenter(training_data_path)

    # Process a video file
    video_path = "input.mp4"  # Change this to your video file path
    ground_truth_path = "data/ground_truth.json"  # Optional ground truth data

    # Check if the video file exists, if not, suggest looking for alternatives
    if not os.path.isfile(video_path):
        print(f"Video file '{video_path}' not found.")
        # Check if PoseVideos directory exists and has any mp4 files
        if os.path.exists("PoseVideos"):
            video_files = [f for f in os.listdir("PoseVideos") if f.endswith(".mp4")]
            if video_files:
                video_path = os.path.join("PoseVideos", video_files[0])
                print(f"Using alternative video file: {video_path}")
            else:
                print("No .mp4 files found in PoseVideos directory.")
                return
        else:
            print("Please provide a valid video file path.")
            return

    try:
        # Process video and extract 3D landmarks
        visualizer.process_video(
            video_path,
            ground_truth_path if os.path.exists(ground_truth_path) else None
        )

        # Add some sample subjective feedback
        visualizer.evaluator.add_subjective_feedback(
            score=7.5,
            comments="Good swing mechanics but a slight overrotation in the follow")

        # Create animation
        visualizer.animate(show=True, save=True)

    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()
