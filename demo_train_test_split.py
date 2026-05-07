"""
Demo Script for Train/Test Split Configuration
---------------------------------------------
This script demonstrates how to use the train_test_split_config module
to train a MediaPipe pose model with different train/test split ratios.
"""

import os
from train_test_split_config import SwingPhaseSegmenter, train_with_different_splits

def main():
    print("MediaPipe Golf Pose Analysis - Train/Test Split Demo")
    print("---------------------------------------------------")
    
    # Path to training data
    training_data_path = "data/swing_phase_labels.csv"
    
    # Check if training data exists
    if not os.path.exists(training_data_path):
        print(f"Training data file {training_data_path} not found.")
        print("Please make sure you have labeled data in the correct format.")
        return
    
    print("\nOption 1: Train with a specific split ratio")
    print("-------------------------------------------")
    print("1. Train with 70-30 split (default)")
    print("2. Train with 90-10 split")
    print("3. Train with 60-40 split")
    print("4. Train with all splits and compare results")
    
    choice = input("\nEnter your choice (1-4) or press Enter for option 4: ")
    
    if choice == "1":
        # Train with 70-30 split (default)
        print("\nTraining with 70-30 split...")
        segmenter = SwingPhaseSegmenter()
        # In a real scenario, you would load your data here
        # For demonstration, we'll use the train_with_different_splits function with just one split
        train_with_specific_split(training_data_path, 0.3)
        
    elif choice == "2":
        # Train with 90-10 split
        print("\nTraining with 90-10 split...")
        train_with_specific_split(training_data_path, 0.1)
        
    elif choice == "3":
        # Train with 60-40 split
        print("\nTraining with 60-40 split...")
        train_with_specific_split(training_data_path, 0.4)
        
    else:
        # Train with all splits and compare
        print("\nTraining with all splits and comparing results...")
        results = train_with_different_splits(training_data_path)
        
        if results:
            print("\nTraining complete! Models saved in the 'models' directory.")
            print("You can now use these models for pose estimation and analysis.")

def train_with_specific_split(training_data_path, test_size):
    """Train with a specific test_size"""
    import pandas as pd
    import numpy as np
    
    try:
        # Load data
        df = pd.read_csv(training_data_path)
        
        # Extract features and labels
        landmark_sequences = []
        labels = []
        
        for _, row in df.iterrows():
            landmarks = np.array(eval(row['landmarks']))
            phase = row['phase']
            landmark_sequences.append(landmarks)
            labels.append(phase)
        
        # Create segmenter and train
        segmenter = SwingPhaseSegmenter()
        accuracy = segmenter.train(landmark_sequences, labels, test_size=test_size)
        
        # Save model
        train_pct = int(100 * (1 - test_size))
        test_pct = int(100 * test_size)
        model_path = f'models/swing_phase_svm_{train_pct}_{test_pct}.pkl'
        segmenter.save_model(model_path)
        
        print(f"\nTraining complete! Model saved as {model_path}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False

if __name__ == "__main__":
    main()