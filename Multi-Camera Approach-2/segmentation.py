"""
Golf Swing Phase Segmentation using SVM/MLP

This module implements machine learning models (SVM/MLP) for segmenting
golf swing sequences into distinct phases.

Research Objectives:
    2. Effectiveness of ML swing-phase segmentation [[44], [47]]
    
Technical Constraints:
    - Provide F1-score for swing phase segmentation
    - Encapsulate hyperparameters at top of file for quick tuning
"""

import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import joblib


# Hyperparameters for segmentation
MODEL_TYPE = "SVM"  # Options: "SVM" or "MLP"
SVM_KERNEL = 'rbf'  # Options: 'linear', 'poly', 'rbf', 'sigmoid'
SVM_C = 1.0  # Regularization parameter
MLP_HIDDEN_LAYERS = (100, 50)  # Hidden layer sizes for MLP
MLP_ACTIVATION = 'relu'  # Options: 'identity', 'logistic', 'tanh', 'relu'
RANDOM_STATE = 42  # For reproducibility
TEST_SIZE = 0.2  # Proportion of data to use for testing


class SwingPhaseSegmenter:
    """
    Golf swing phase segmentation using machine learning (SVM/MLP).
    
    Research Alignment:
        Supports Objective 2: Effectiveness of ML swing-phase segmentation [[44], [47]]
        by implementing and evaluating SVM and MLP models for phase detection.
    """

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

    def __init__(self, model_path=None, model_type=MODEL_TYPE):
        """
        Initialize the swing phase segmenter.
        
        Args:
            model_path: Path to a pre-trained model file (if None, creates a new model)
            model_type: Type of model to use ("SVM" or "MLP")
        """
        self.scaler = StandardScaler()
        self.model_type = model_type
        
        # Try to load pre-trained model if it exists
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Loaded {model_type} model from {model_path}")
        else:
            # Create a new model based on specified type
            if model_type == "SVM":
                self.model = svm.SVC(
                    kernel=SVM_KERNEL,
                    C=SVM_C,
                    decision_function_shape='ovr',
                    random_state=RANDOM_STATE
                )
            elif model_type == "MLP":
                self.model = MLPClassifier(
                    hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                    activation=MLP_ACTIVATION,
                    random_state=RANDOM_STATE
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}. Use 'SVM' or 'MLP'.")
                
            print(f"Created new {model_type} model for swing phase segmentation")

    def extract_features(self, landmarks):
        """
        Extract features from pose landmarks that are relevant for swing phase detection.
        
        Args:
            landmarks: Array of shape (33, 3) with x,y,z coordinates for pose landmarks
            
        Returns:
            Feature vector
            
        Research Alignment:
            Feature extraction based on biomechanical principles from [[44]].
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
        
        # Hip to shoulder angle (spine angle)
        spine_vector = (l_shoulder + r_shoulder) / 2 - (l_hip + r_hip) / 2
        spine_angle = np.arctan2(spine_vector[1], spine_vector[0]) * 180 / np.pi
        features.append(spine_angle)
        
        # Knee flexion (both knees)
        l_hip_knee_ankle_angle = self._calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_hip_knee_ankle_angle = self._calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        features.append(l_hip_knee_ankle_angle)
        features.append(r_hip_knee_ankle_angle)

        return np.array(features)

    def _calculate_angle(self, a, b, c):
        """
        Calculate angle between three 3D points.
        
        Args:
            a, b, c: 3D points where b is the vertex
            
        Returns:
            Angle in degrees
        """
        ba = a - b
        bc = c - b

        # Calculate dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def _angle_between_vectors(self, v1, v2):
        """
        Calculate the angle between two vectors.
        
        Args:
            v1, v2: Vectors to calculate angle between
            
        Returns:
            Angle in degrees
        """
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)

        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    def train(self, landmark_sequences, labels):
        """
        Train the model on labeled sequences.
        
        Args:
            landmark_sequences: List of landmark arrays
            labels: Corresponding phase labels for each sequence
            
        Returns:
            Dictionary with training metrics (accuracy, recall, f1)
            
        Research Alignment:
            Evaluation metrics follow [[47]] for comparing segmentation performance.
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
            scaled_features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print metrics
        print(f"Training complete - Accuracy: {accuracy:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        return {
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }

    def predict(self, landmarks):
        """
        Predict the swing phase for a single frame.
        
        Args:
            landmarks: Array of shape (33, 3) with pose landmarks
            
        Returns:
            Predicted phase as string
        """
        features = self.extract_features(landmarks)
        scaled_features = self.scaler.transform([features])
        phase_id = self.model.predict(scaled_features)[0]
        return self.PHASES[phase_id]
    
    def predict_sequence(self, landmark_sequence):
        """
        Predict phases for a sequence of frames.
        
        Args:
            landmark_sequence: List of landmark arrays
            
        Returns:
            List of predicted phases
        """
        predictions = []
        for landmarks in landmark_sequence:
            predictions.append(self.predict(landmarks))
        return predictions

    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # Also save the scaler
        scaler_path = os.path.splitext(model_path)[0] + "_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")


def train_from_csv(csv_path, model_type=MODEL_TYPE, output_path=None):
    """
    Train a segmentation model from a CSV file containing labeled data.
    
    Args:
        csv_path: Path to CSV file with columns for landmarks and phase labels
        model_type: Type of model to use ("SVM" or "MLP")
        output_path: Path to save the trained model
        
    Returns:
        Trained SwingPhaseSegmenter instance
    """
    # Load data from CSV
    data = pd.read_csv(csv_path)
    
    # Extract landmarks and labels
    # This assumes the CSV has columns for landmarks and a 'phase' column
    # The exact format will depend on your data
    
    # Example extraction (modify based on your CSV format):
    landmark_columns = [col for col in data.columns if col.startswith('landmark_')]
    landmarks = data[landmark_columns].values.reshape(-1, 33, 3)
    labels = data['phase'].values
    
    # Create and train segmenter
    segmenter = SwingPhaseSegmenter(model_type=model_type)
    metrics = segmenter.train(landmarks, labels)
    
    # Save model if output path is provided
    if output_path:
        segmenter.save_model(output_path)
    
    # Print F1 score as required by technical constraints
    print(f"Swing-phase F1-score: {metrics['f1']:.4f}")
    
    return segmenter


if __name__ == "__main__":
    # Example usage
    print("Swing Phase Segmentation module - Run through the main pipeline")