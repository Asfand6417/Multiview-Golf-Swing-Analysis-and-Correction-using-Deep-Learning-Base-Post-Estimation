import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import cv2
import mediapipe as mp

# Define constants
LANDMARKS_COUNT = 33  # MediaPipe provides 33 landmarks
FEATURES_PER_LANDMARK = 3  # x, y, z coordinates
MAX_FRAMES = 64  # Maximum number of frames to process per video
MODEL_DIR = 'models'
SWING_PHASES = ["Address", "Takeaway", "Backswing", "Top of Backswing", 
                "Downswing", "Impact", "Follow Through"]

def create_lstm_model(input_shape, num_classes):
    """
    Create a Bidirectional LSTM model for pose sequence classification
    
    Args:
        input_shape: Shape of input data (frames, features)
        num_classes: Number of output classes (swing phases)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Bidirectional LSTM for sequence learning
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_pose_error_model(input_shape):
    """
    Create a regression model for predicting pose alignment error (MPJPE)
    
    Args:
        input_shape: Shape of input data (frames, features)
        
    Returns:
        Compiled Keras model
    """
    input_layer = Input(shape=input_shape)
    
    x = LSTM(128, return_sequences=True)(input_layer)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='linear')(x)  # Regression output (MPJPE score)
    
    model = Model(inputs=input_layer, outputs=x)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def extract_pose_features_from_video(video_path, sample_rate=3):
    """
    Extract MediaPipe pose landmarks from video frames
    
    Args:
        video_path: Path to video file
        sample_rate: Sample every Nth frame
        
    Returns:
        List of frame landmarks in format [frames, landmarks, coordinates]
    """
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            frame_count += 1
            if frame_count % sample_rate != 0:
                continue
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = pose.process(image)
            
            # Extract landmarks
            if results.pose_world_landmarks:
                landmarks = []
                for landmark in results.pose_world_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                frames.append(landmarks)
    
    cap.release()
    return frames

def extract_pose_features_from_landmarks(landmarks_list):
    """
    Extract features from pose detector landmark list
    
    Args:
        landmarks_list: List of landmarks from MediaPipe detector [id, x, y]
        
    Returns:
        Feature vector for model input
    """
    features = []
    
    # Convert from [id, x, y] format to just coordinates
    for lm in landmarks_list:
        if len(lm) > 2:  # Handle [id, x, y] format
            features.extend([lm[1], lm[2], 0])  # Add z=0 for 2D landmarks
    
    return features

def preprocess_video_dataset(video_paths, labels=None, max_frames=MAX_FRAMES):
    """
    Preprocess multiple videos for model training
    
    Args:
        video_paths: List of paths to video files
        labels: Optional list of labels for videos
        max_frames: Maximum number of frames to use per video
        
    Returns:
        X: Feature data (pose landmarks)
        y: Labels if provided
    """
    X = []
    
    for video_path in video_paths:
        # Extract pose features
        pose_sequence = extract_pose_features_from_video(video_path)
        
        if len(pose_sequence) < 10:  # Skip if too few frames detected
            continue
            
        # Pad or truncate sequence to fixed length
        if len(pose_sequence) > max_frames:
            # Sample frames evenly
            indices = np.linspace(0, len(pose_sequence)-1, max_frames, dtype=int)
            pose_sequence = [pose_sequence[i] for i in indices]
        else:
            # Pad with zeros
            padding = [np.zeros(LANDMARKS_COUNT * FEATURES_PER_LANDMARK) 
                      for _ in range(max_frames - len(pose_sequence))]
            pose_sequence = pose_sequence + padding
            
        X.append(pose_sequence)
    
    if labels is not None:
        return np.array(X), np.array(labels)
    else:
        return np.array(X)

def train_lstm_classifier(X_train, y_train, X_val, y_val, num_classes=len(SWING_PHASES)):
    """
    Train an LSTM model for golf swing phase classification
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        num_classes: Number of output classes
        
    Returns:
        Trained model and training history
    """
    # Ensure output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create and train the model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (frames, features)
    
    model = create_lstm_model(input_shape, num_classes)
    print(model.summary())
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(f'{MODEL_DIR}/golf_swing_phase_lstm.h5', 
                       monitor='val_loss', save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
    return model, history

def train_mpjpe_model(X_train, y_mpjpe_train, X_val, y_mpjpe_val):
    """
    Train a regression model for MPJPE (pose error) prediction
    
    Args:
        X_train: Training features
        y_mpjpe_train: Training MPJPE values
        X_val: Validation features
        y_mpjpe_val: Validation MPJPE values
        
    Returns:
        Trained model and training history
    """
    # Ensure output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create and train the model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (frames, features)
    
    model = create_pose_error_model(input_shape)
    print(model.summary())
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(f'{MODEL_DIR}/golf_swing_mpjpe_lstm.h5', 
                       monitor='val_loss', save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_mpjpe_train,
        validation_data=(X_val, y_mpjpe_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks
    )
    
    return model, history

def evaluate_phase_model(model, X_test, y_test):
    """
    Evaluate the trained swing phase classification model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("Golf Swing Phase Classification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

def evaluate_mpjpe_model(model, X_test, y_mpjpe_test):
    """
    Evaluate the trained MPJPE regression model
    
    Args:
        model: Trained model
        X_test: Test features
        y_mpjpe_test: Test MPJPE values
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_mpjpe_pred = model.predict(X_test).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(y_mpjpe_test - y_mpjpe_pred))
    rmse = np.sqrt(np.mean((y_mpjpe_test - y_mpjpe_pred)**2))
    
    # Print results
    print("\nMPJPE Regression Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'y_pred': y_mpjpe_pred
    }

def plot_training_history(history, model_type='classification'):
    """
    Plot training history
    
    Args:
        history: Training history from model.fit()
        model_type: 'classification' or 'regression'
    """
    plt.figure(figsize=(12, 4))
    
    if model_type == 'classification':
        # Accuracy subplot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Loss subplot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    else:
        # MAE subplot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Loss subplot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/training_history_{model_type}.png')
    plt.show()

def plot_confusion_matrix(cm, class_names=SWING_PHASES):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix from evaluate_phase_model()
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{MODEL_DIR}/confusion_matrix.png')
    plt.show()

def predict_swing_phase(model, landmarks_list, preprocess=True):
    """
    Predict swing phase from a single frame's landmarks
    
    Args:
        model: Trained LSTM model
        landmarks_list: List of landmarks from MediaPipe detector
        preprocess: Whether to preprocess the landmarks
        
    Returns:
        Predicted phase name and confidence
    """
    if preprocess:
        # Extract features
        features = extract_pose_features_from_landmarks(landmarks_list)
    else:
        # Assume already preprocessed
        features = np.array(landmarks_list)
    
    # Create proper sequence structure for LSTM
    sequence = create_sequence_from_single_frame(features)
    
    # Get predictions
    predictions = model.predict(sequence)[0]
    phase_idx = np.argmax(predictions)
    confidence = float(predictions[phase_idx])
    
    return SWING_PHASES[phase_idx], confidence

def predict_mpjpe(model, landmarks_list, preprocess=True):
    """
    Predict MPJPE (pose error) from a single frame's landmarks
    
    Args:
        model: Trained MPJPE model
        landmarks_list: List of landmarks from MediaPipe detector
        preprocess: Whether to preprocess the landmarks
        
    Returns:
        Predicted MPJPE value
    """
    if preprocess:
        # Extract features
        features = extract_pose_features_from_landmarks(landmarks_list)
        
        # Reshape for model input (add batch and sequence dimensions)
        X = np.array([[features]])
    else:
        # Assume already preprocessed but still need proper dimensions
        features = np.array(landmarks_list)
        sequence = np.zeros((1, MAX_FRAMES, features.shape[0]))
        sequence[0, 0, :] = features
    
    # Get predictions
    mpjpe = float(model.predict(X)[0][0])
    
    return mpjpe

def create_sequence_from_single_frame(features):
    """
    Create a properly shaped sequence from a single frame's features for LSTM prediction
    
    Args:
        features: Feature vector from a single frame
        
    Returns:
        Properly shaped sequence for LSTM input
    """
    # LSTM expects input shape: [batch_size, sequence_length, features]
    sequence = np.zeros((1, MAX_FRAMES, len(features)))
    sequence[0, 0, :] = features  # Add features to first frame of sequence
    return sequence

def load_trained_models():
    """
    Load previously trained models
    
    Returns:
        Dictionary with loaded models
    """
    models = {}
    
    # Check if phase classification model exists
    phase_model_path = f'{MODEL_DIR}/golf_swing_phase_lstm.h5'
    if os.path.exists(phase_model_path):
        models['phase'] = load_model(phase_model_path)
        print(f"Loaded phase classification model from {phase_model_path}")
    
    # Check if MPJPE regression model exists
    mpjpe_model_path = f'{MODEL_DIR}/golf_swing_mpjpe_lstm.h5'
    if os.path.exists(mpjpe_model_path):
        models['mpjpe'] = load_model(mpjpe_model_path)
        print(f"Loaded MPJPE regression model from {mpjpe_model_path}")
    
    return models

# Usage example for the full training pipeline
def train_full_pipeline(video_dir, annotations_file):
    """
    Run the full training pipeline
    
    Args:
        video_dir: Directory containing video files
        annotations_file: Path to CSV file with annotations
    """
    # Load annotations
    annotations = pd.read_csv(annotations_file)
    print(f"Loaded {len(annotations)} annotated golf swings")
    
    # Prepare data
    video_paths = [os.path.join(video_dir, f"{row['video_id']}.mp4") 
                 for _, row in annotations.iterrows()]
    phase_labels = annotations['phase'].values
    mpjpe_values = annotations['mpjpe'].values if 'mpjpe' in annotations.columns else None
    
    # Split data
    video_train, video_test, y_train, y_test = train_test_split(
        video_paths, phase_labels, test_size=0.3, random_state=42, stratify=phase_labels)
    
    video_val, video_test, y_val, y_test = train_test_split(
        video_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    
    print(f"Training set: {len(video_train)} videos")
    print(f"Validation set: {len(video_val)} videos")
    print(f"Testing set: {len(video_test)} videos")
    
    # Process videos
    print("Processing training videos...")
    X_train, y_train = preprocess_video_dataset(video_train, y_train)
    print("Processing validation videos...")
    X_val, y_val = preprocess_video_dataset(video_val, y_val)
    print("Processing test videos...")
    X_test, y_test = preprocess_video_dataset(video_test, y_test)
    
    # Train phase classification model
    print("\nTraining swing phase classification model...")
    phase_model, phase_history = train_lstm_classifier(X_train, y_train, X_val, y_val)
    
    # Evaluate phase model
    print("\nEvaluating swing phase classification model...")
    phase_results = evaluate_phase_model(phase_model, X_test, y_test)
    
    # Plot training history and confusion matrix
    plot_training_history(phase_history, 'classification')
    plot_confusion_matrix(phase_results['confusion_matrix'])
    
    # Train MPJPE model if data is available
    if mpjpe_values is not None:
        # Split MPJPE values
        mpjpe_train = mpjpe_values[:len(y_train)]
        mpjpe_val = mpjpe_values[len(y_train):len(y_train) + len(y_val)]
        mpjpe_test = mpjpe_values[len(y_train) + len(y_val):]
        
        print("\nTraining MPJPE regression model...")
        mpjpe_model, mpjpe_history = train_mpjpe_model(
            X_train, mpjpe_train, X_val, mpjpe_val)
        
        # Evaluate MPJPE model
        print("\nEvaluating MPJPE regression model...")
        mpjpe_results = evaluate_mpjpe_model(mpjpe_model, X_test, mpjpe_test)
        
        # Plot training history
        plot_training_history(mpjpe_history, 'regression')
        
        print("\nTraining complete! Models saved to:", MODEL_DIR)
    else:
        print("\nMPJPE data not available. Skipping MPJPE model training.")
        print("\nPartial training complete! Phase model saved to:", MODEL_DIR)

if __name__ == "__main__":
    # Example usage:
    # Change these paths to your actual data locations
    print("LSTM model builder loaded. Run train_full_pipeline() to train models.")