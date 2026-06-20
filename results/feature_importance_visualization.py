import matplotlib.pyplot as plt
import numpy as np

# Simulated feature importance from SVM model
features = [
    "Left Elbow Angle", 
    "Right Elbow Angle", 
    "Spine Angle", 
    "Wrist Height", 
    "Left Knee Angle", 
    "Right Knee Angle",
    "Shoulder Alignment",
    "Hip Alignment",
    "Wrist Position X",
    "Wrist Position Y",
    "Torso Rotation"
]

# Simulated importance values (would come from your SVM model)
importance = [0.18, 0.15, 0.14, 0.12, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

# Sort in descending order
sorted_indices = np.argsort(importance)[::-1]
sorted_features = [features[i] for i in sorted_indices]
sorted_importance = [importance[i] for i in sorted_indices]

# Create horizontal bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(sorted_features, sorted_importance, color='teal')

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
             f"{sorted_importance[i]:.3f}", va='center')

plt.xlabel('Importance Score')
plt.title('Feature Importance for Golf Swing Phase Classification')
plt.grid(axis='x', alpha=0.3)
plt.xlim(0, max(importance) * 1.2)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()