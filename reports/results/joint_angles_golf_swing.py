import matplotlib.pyplot as plt
import numpy as np

# Simulated data based on your project
phases = ["Address", "Takeaway", "Backswing", "Top", "Downswing", "Impact", "Follow-through"]
frame_indices = [0, 15, 30, 45, 60, 75, 90]  # Example frame indices for each phase

# Sample joint angles across frames (would be calculated from your landmarks)
left_elbow_angles = [165, 155, 120, 90, 130, 160, 170]
right_elbow_angles = [168, 170, 145, 100, 95, 130, 150]
spine_angles = [160, 155, 150, 145, 150, 155, 165]
left_knee_angles = [170, 168, 165, 165, 168, 172, 175]
right_knee_angles = [172, 170, 168, 168, 165, 168, 170]

plt.figure(figsize=(12, 8))
plt.plot(frame_indices, left_elbow_angles, 'r-', marker='o', label='Left Elbow Angle')
plt.plot(frame_indices, right_elbow_angles, 'b-', marker='s', label='Right Elbow Angle')
plt.plot(frame_indices, spine_angles, 'g-', marker='^', label='Spine Angle')
plt.plot(frame_indices, left_knee_angles, 'm-', marker='*', label='Left Knee Angle')
plt.plot(frame_indices, right_knee_angles, 'c-', marker='d', label='Right Knee Angle')

# Add phase annotations with horizontal text
for i, phase in enumerate(phases):
    plt.axvline(x=frame_indices[i], color='gray', linestyle='--', alpha=0.5)

# Set x-ticks and labels for phases (horizontally)
plt.xticks(frame_indices, phases)

plt.grid(True, alpha=0.3)
plt.xlabel('Swing Phase')
plt.ylabel('Angle (degrees)')
plt.title('Joint Angles Throughout Golf Swing Phases')
plt.legend(loc='lower right')

# Adjust bottom margin to ensure labels fit
plt.tight_layout()
plt.savefig('joint_angles_throughout_swing.png', dpi=300)
plt.show()