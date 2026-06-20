import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Simulated MPJPE data for different body parts
joint_groups = {
    "Arms": ["Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist"],
    "Legs": ["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"],
    "Torso": ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"],
    "Head": ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear"]
}

# Simulated error values for each joint (would come from your evaluator)
joint_errors = {
    "Left Shoulder": [0.028, 0.032, 0.025, 0.031, 0.029],
    "Right Shoulder": [0.031, 0.035, 0.028, 0.030, 0.033],
    "Left Elbow": [0.041, 0.039, 0.044, 0.038, 0.042],
    "Right Elbow": [0.040, 0.045, 0.042, 0.038, 0.041],
    "Left Wrist": [0.052, 0.048, 0.055, 0.050, 0.053],
    "Right Wrist": [0.050, 0.058, 0.054, 0.049, 0.055],
    "Left Hip": [0.025, 0.028, 0.023, 0.026, 0.024],
    "Right Hip": [0.026, 0.029, 0.025, 0.027, 0.028],
    "Left Knee": [0.035, 0.038, 0.033, 0.036, 0.035],
    "Right Knee": [0.037, 0.039, 0.035, 0.038, 0.036],
    "Left Ankle": [0.048, 0.051, 0.046, 0.049, 0.047],
    "Right Ankle": [0.049, 0.052, 0.047, 0.050, 0.048],
    "Nose": [0.018, 0.021, 0.017, 0.020, 0.019],
    "Left Eye": [0.019, 0.022, 0.018, 0.021, 0.020],
    "Right Eye": [0.020, 0.023, 0.019, 0.022, 0.021],
    "Left Ear": [0.022, 0.025, 0.020, 0.023, 0.021],
    "Right Ear": [0.023, 0.026, 0.021, 0.024, 0.022]
}

# Prepare data for box plot
data = []
labels = []
colors = []
group_colors = {'Arms': 'skyblue', 'Legs': 'lightgreen', 'Torso': 'salmon', 'Head': 'purple'}

for group, joints in joint_groups.items():
    for joint in joints:
        if joint in joint_errors:
            data.append(joint_errors[joint])
            labels.append(joint)
            colors.append(group_colors[group])

# Create figure
plt.figure(figsize=(16, 10))

# Create boxplot
box = plt.boxplot(data, patch_artist=True, labels=labels, showfliers=False)

# Color boxes according to group
for box_patch, color in zip(box['boxes'], colors):
    box_patch.set_facecolor(color)

# Customize boxplot
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylabel('MPJPE (meters)')
plt.title('Mean Per Joint Position Error Distribution by Body Part')

# Add a legend for groups
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', alpha=0.7, label=group)
                  for group, color in group_colors.items()]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('mpjpe_distribution.png', dpi=300)
plt.show()