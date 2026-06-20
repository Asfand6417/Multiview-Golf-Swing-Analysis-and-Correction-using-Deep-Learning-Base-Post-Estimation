# MediaPipe Train/Test Split Configuration

This module provides functionality to train MediaPipe pose models with configurable train/test split ratios for golf swing analysis.

## Overview

MediaPipe is a pre-trained model framework that can be used for pose detection and analysis. This implementation allows you to:

1. Train a golf swing phase segmentation model with different train/test split ratios:
   - 70-30 split (default)
   - 90-10 split
   - 60-40 split

2. Compare the performance of models trained with different split ratios to determine the optimal configuration for your specific use case.

## Files

- `src/analysis/train_test_split_config.py`: Contains the modified `SwingPhaseSegmenter` class with configurable train/test split functionality
- `src/analysis/demo_train_test_split.py`: A demo script that shows how to use the train/test split configuration

## Usage

### Basic Usage

```python
from src.analysis.train_test_split_config import SwingPhaseSegmenter

# Create a segmenter with default settings
segmenter = SwingPhaseSegmenter()

# Train with 70-30 split (default)
segmenter.train(landmark_sequences, labels)

# Train with 90-10 split
segmenter.train(landmark_sequences, labels, test_size=0.1)

# Train with 60-40 split
segmenter.train(landmark_sequences, labels, test_size=0.4)
```

### Training with Different Splits and Comparing Results

```python
from src.analysis.train_test_split_config import train_with_different_splits

# Train with all splits and compare results
results = train_with_different_splits("data/swing_phase_labels.csv")
```

### Using the Demo Script

1. Run the demo script:
   ```
   python src/analysis/demo_train_test_split.py
   ```

2. Follow the prompts to select a train/test split ratio or compare all ratios.

## Implementation Details

The `SwingPhaseSegmenter` class in `src/analysis/train_test_split_config.py` is a modified version of the original class from `src/analysis/main.py`. The key modification is in the `train` method, which now accepts a `test_size` parameter to configure the train/test split ratio.

```python
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
    # ... implementation ...
```

The `train_with_different_splits` function trains the model with different split ratios and compares the results:

```python
def train_with_different_splits(training_data_path):
    """
    Train the model with different train/test splits and compare results
    
    Args:
        training_data_path: Path to a CSV file with labeled swing phases
    """
    # ... implementation ...
```

## Data Format

The training data should be in a CSV file with at least two columns:
- `landmarks`: A string representation of a numpy array containing the pose landmarks
- `phase`: The corresponding swing phase label

## Model Output

Trained models are saved in the `models` directory with filenames that indicate the train/test split ratio:
- `swing_phase_svm_70_30.pkl`: Model trained with 70-30 split
- `swing_phase_svm_90_10.pkl`: Model trained with 90-10 split
- `swing_phase_svm_60_40.pkl`: Model trained with 60-40 split
