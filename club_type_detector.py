"""
Club Type Detector for Golf Swing Analysis
------------------------------------------
This module provides functionality to detect whether a golf swing is a Driver or Iron shot
based on pose landmarks from MediaPipe.
"""

import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ClubTypeDetector:
    """Detector for classifying golf swings as Driver or Iron shots"""
    
    # Define club types
    CLUB_TYPES = {
        0: "Driver",
        1: "Iron"
    }
    
    def __init__(self, model_path=None):
        """Initialize the club type detector"""
        self.scaler = StandardScaler()
        
        # Try to load pre-trained model if it exists
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Loaded club type detection model from {model_path}")
        else:
            # Create a new SVM model
            self.model = svm.SVC(kernel='rbf', decision_function_shape='ovr')
            print("Created new SVM model for club type detection")
            
            # Initialize with some default parameters
            # These would ideally be trained on labeled data
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize model with default parameters for basic detection"""
        # This is a simplified approach - in a real implementation,
        # you would train the model on labeled data
        pass
    
    def extract_features(self, landmarks):
        """
        Extract features from pose landmarks that are relevant for club type detection
        
        Args:
            landmarks: Array of shape (33, 3) with x,y,z coordinates for MediaPipe pose landmarks
            
        Returns:
            Feature vector
        """
        features = []
        
        # Extract key joint positions
        # Shoulders
        r_shoulder = landmarks[12]
        l_shoulder = landmarks[11]
        
        # Hips
        r_hip = landmarks[24]
        l_hip = landmarks[23]
        
        # Feet
        r_foot = landmarks[30]  # Right foot index
        l_foot = landmarks[29]  # Left foot index
        
        # Feature 1: Stance width (distance between feet)
        # Driver typically has wider stance than Iron
        stance_width = np.linalg.norm(r_foot[:2] - l_foot[:2])  # Using only x,z coordinates (horizontal plane)
        features.append(stance_width)
        
        # Feature 2: Spine angle (vertical vs. bent over)
        # Iron shots typically have more bent spine angle
        mid_shoulder = (r_shoulder + l_shoulder) / 2
        mid_hip = (r_hip + l_hip) / 2
        spine_vector = mid_shoulder - mid_hip
        vertical_vector = np.array([0, 1, 0])  # Y is up
        spine_angle = self._angle_between_vectors(spine_vector, vertical_vector)
        features.append(spine_angle)
        
        # Feature 3: Shoulder tilt
        # Driver typically has more shoulder tilt than Iron
        shoulder_vector = r_shoulder - l_shoulder
        horizontal_vector = np.array([1, 0, 0])  # X is horizontal
        shoulder_tilt = self._angle_between_vectors(shoulder_vector[:2], horizontal_vector[:2])
        features.append(shoulder_tilt)
        
        # Feature 4: Ball position (approximated by front foot position relative to center)
        # Driver typically has ball positioned more forward (toward target)
        center = (r_foot + l_foot) / 2
        ball_position = l_foot[2] - center[2]  # Z coordinate (forward/backward)
        features.append(ball_position)
        
        # Feature 5: Hip rotation
        # Driver typically has more hip rotation
        hip_vector = r_hip - l_hip
        hip_rotation = self._angle_between_vectors(hip_vector[:2], horizontal_vector[:2])
        features.append(hip_rotation)
        
        return np.array(features)
    
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
            labels: Corresponding club type labels for each sequence
            
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
        
        print(f"Club type detection training complete - Accuracy: {accuracy:.2f}")
        
        return accuracy
    
    def predict(self, landmarks):
        """
        Predict the club type for a single frame
        
        Args:
            landmarks: Array of shape (33, 3) with pose landmarks
            
        Returns:
            Predicted club type as string ("Driver" or "Iron")
        """
        features = self.extract_features(landmarks)
        
        # If model is not trained, make a simple heuristic prediction
        if not hasattr(self.model, 'support_'):
            # Simple heuristic: if stance width is wide, predict Driver, otherwise Iron
            stance_width = features[0]
            spine_angle = features[1]
            
            if stance_width > 0.3 and spine_angle < 30:
                return "Driver"
            else:
                return "Iron"
        
        # Use trained model for prediction
        scaled_features = self.scaler.transform([features])
        club_type_id = self.model.predict(scaled_features)[0]
        return self.CLUB_TYPES[club_type_id]
    
    def save_model(self, model_path):
        """Save the trained model to disk"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Club type detection model saved to {model_path}")