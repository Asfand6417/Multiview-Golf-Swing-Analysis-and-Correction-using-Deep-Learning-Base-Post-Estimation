import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Simulated joint angle data
joints = ["Left Elbow", "Right Elbow", "Spine", "Left Knee", "Right Knee"]

# Expert recommended ranges
expert_min = [160, 160, 145, 165, 165]
expert_max = [190, 190, 165, 180, 180]

# Measured angles for different skill levels
pro_angles = [175, 172, 155, 170, 173]
amateur_angles = [158, 162, 142, 164, 168]
beginner_angles = [145, 149, 138, 155, 162]

# Acceptable range
pro_std = [5, 5, 3, 3, 3]
amateur_std = [8, 10, 7, 5, 5]
beginner_std = [12, 15, 10, 8, 9]

fig, ax = plt.subplots(figsize=(14, 8))

# X positions
x = np.arange(len(joints))
width = 0.2

# Plot bars
rect1 = ax.bar(x - width, pro_angles, width, label='Pro Golfer', color='green', alpha=0.7)
rect2 = ax.bar(x, amateur_angles, width, label='Amateur', color='orange', alpha=0.7)
rect3 = ax.bar(x + width, beginner_angles, width, label='Beginner', color='red', alpha=0.7)

# Add error bars
ax.errorbar(x - width, pro_angles, yerr=pro_std, fmt='none', color='black', capsize=5)
ax.errorbar(x, amateur_angles, yerr=amateur_std, fmt='none', color='black', capsize=5)
ax.errorbar(x + width, beginner_angles, yerr=beginner_std, fmt='none', color='black', capsize=5)

# Plot recommended range as rectangles
for i in range(len(joints)):
    ax.add_patch(plt.Rectangle((i - 0.4, expert_min[i]), 0.8, expert_max[i] - expert_min[i], 
                              fill=True, alpha=0.2, color='blue'))

# Add labels and title
ax.set_ylabel('Angle (degrees)')
ax.set_title('Joint Angle Comparison: Pro vs Amateur vs Beginner Golfers')
ax.set_xticks(x)
ax.set_xticklabels(joints)
ax.legend()

# Add text label for expert range
ax.text(len(joints)-1.5, expert_min[-1] - 10, "Expert Recommended Range", fontsize=10, color='blue')

# Grid for readability
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('joint_angle_comparison.png', dpi=300)
plt.show()