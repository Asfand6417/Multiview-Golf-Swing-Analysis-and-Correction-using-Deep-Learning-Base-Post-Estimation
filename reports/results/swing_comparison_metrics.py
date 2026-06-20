import matplotlib.pyplot as plt
import numpy as np

# Simulated evaluation data for 3 different golf swings
swing_ids = ["Pro Golfer", "Amateur", "Beginner"]
mpjpe_values = [0.032, 0.081, 0.127]  # Lower is better
phase_accuracy = [0.91, 0.76, 0.58]   # Higher is better
subjective_scores = [8.5, 6.2, 4.1]   # Higher is better
per_joint_error = {
    "Left Elbow": [0.025, 0.065, 0.112],
    "Right Elbow": [0.031, 0.072, 0.118],
    "Left Shoulder": [0.028, 0.060, 0.093],
    "Right Shoulder": [0.030, 0.068, 0.102],
    "Spine": [0.041, 0.092, 0.148]
}

# Create grouped bar chart
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# Plot MPJPE values (lower is better)
axes[0].bar(swing_ids, mpjpe_values, color=['green', 'orange', 'red'])
axes[0].set_ylabel('MPJPE (meters)')
axes[0].set_title('Mean Per Joint Position Error (lower is better)')
axes[0].grid(axis='y', alpha=0.3)
# Add value labels
for i, v in enumerate(mpjpe_values):
    axes[0].text(i, v + 0.005, f"{v:.3f}", ha='center')

# Plot phase accuracy (higher is better)
axes[1].bar(swing_ids, phase_accuracy, color=['green', 'orange', 'red'])
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Phase Detection Accuracy (higher is better)')
axes[1].set_ylim(0, 1.0)
axes[1].grid(axis='y', alpha=0.3)
# Add percentage labels
for i, v in enumerate(phase_accuracy):
    axes[1].text(i, v + 0.02, f"{v*100:.1f}%", ha='center')

# Plot per-joint errors
x = np.arange(len(swing_ids))
width = 0.15
multiplier = 0

for joint, errors in per_joint_error.items():
    offset = width * multiplier
    rects = axes[2].bar(x + offset, errors, width, label=joint)
    multiplier += 1

axes[2].set_ylabel('Error (meters)')
axes[2].set_title('Error by Joint Type')
axes[2].set_xticks(x + width * (len(per_joint_error) - 1) / 2)
axes[2].set_xticklabels(swing_ids)
axes[2].legend(loc='upper right')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('swing_comparison_metrics.png', dpi=300)
plt.show()