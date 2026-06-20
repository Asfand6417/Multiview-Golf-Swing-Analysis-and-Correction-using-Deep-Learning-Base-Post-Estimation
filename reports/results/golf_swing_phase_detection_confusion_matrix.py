import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Simulated confusion matrix for swing phase detection
phases = ["Address", "Takeaway", "Backswing", "Top", "Downswing", "Impact", "Follow-through"]
confusion = np.array([
    [45, 3, 0, 0, 0, 0, 0],    # Address
    [2, 40, 5, 0, 0, 0, 0],    # Takeaway
    [0, 3, 38, 4, 0, 0, 0],    # Backswing
    [0, 0, 2, 41, 5, 0, 0],    # Top
    [0, 0, 0, 3, 35, 4, 0],    # Downswing
    [0, 0, 0, 0, 6, 42, 2],    # Impact
    [0, 0, 0, 0, 0, 3, 44]     # Follow-through
])

# Calculate metrics
accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=phases, yticklabels=phases)
plt.xlabel('Predicted Phase')
plt.ylabel('True Phase')
plt.title(f'Confusion Matrix for Golf Swing Phase Detection\nOverall Accuracy: {accuracy:.2%}')

# Add class accuracies
for i, acc in enumerate(class_accuracy):
    plt.text(-0.5, i + 0.5, f"{acc:.1%}", verticalalignment='center', fontsize=10)

plt.tight_layout()
plt.savefig('phase_detection_confusion_matrix.png', dpi=300)
plt.show()