"""
Train/Test Split Configuration for MediaPipe Golf Pose Analysis
--------------------------------------------------------------
This module provides a modified SwingPhaseSegmenter class with configurable
train/test split ratios for the MediaPipe golf pose analysis project.

Usage:
    from train_test_split_config import SwingPhaseSegmenter
    
    # Create segmenter with default 70-30 split
    segmenter = SwingPhaseSegmenter()
    
    # Train with 90-10 split
    segmenter.train(landmark_sequences, labels, test_size=0.1)
    
    # Train with 60-40 split
    segmenter.train(landmark_sequences, labels, test_size=0.4)
"""

import os
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import joblib

class SwingPhaseSegmenter:
    """Golf swing phase segmentation using SVM with configurable train/test split"""

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

    def train(self, landmark_sequences, labels, test_size=0.3):
        """
        Train the SVM model on labeled sequences with configurable train/test split

        Args:
            landmark_sequences: List of landmark arrays
            labels: Corresponding phase labels for each sequence
            test_size: Proportion of the dataset to include in the test split (default: 0.3)
                       Supported values: 0.3 (70-30 split), 0.1 (90-10 split), 0.4 (60-40 split)

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
            scaled_features, labels, test_size=test_size, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')

        print(f"Training complete - Accuracy: {accuracy:.2f}, Recall: {recall:.2f}")
        print(f"Train/Test Split: {int(100 * (1 - test_size))}-{int(100 * test_size)}")

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

def train_with_different_splits(training_data_path):
    """
    Train the model with different train/test splits and compare results
    
    Args:
        training_data_path: Path to a CSV file with labeled swing phases
    """
    import pandas as pd
    
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
        
        # Train with different splits
        splits = [0.3, 0.1, 0.4]  # 70-30, 90-10, 60-40
        results = {}
        
        for split in splits:
            print(f"\nTraining with {int(100 * (1 - split))}-{int(100 * split)} split:")
            segmenter = SwingPhaseSegmenter()
            accuracy = segmenter.train(landmark_sequences, labels, test_size=split)
            
            # Save model
            model_path = f'models/swing_phase_svm_{int(100 * (1 - split))}_{int(100 * split)}.pkl'
            segmenter.save_model(model_path)
            
            results[f"{int(100 * (1 - split))}-{int(100 * split)}"] = accuracy
        
        # Print comparison
        print("\nResults comparison:")
        for split, accuracy in results.items():
            print(f"Split {split}: Accuracy = {accuracy:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error training phase segmenter: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    training_data_path = "data/swing_phase_labels.csv"
    if os.path.exists(training_data_path):
        print("Training swing phase segmentation model with different splits...")
        train_with_different_splits(training_data_path)
    else:
        print(f"Training data file {training_data_path} not found.")