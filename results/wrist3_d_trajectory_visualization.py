import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Simulated 3D trajectory of right wrist during golf swing
frames = np.arange(0, 100, 1)
x_coords = 0.5 * np.sin(frames * 0.1) - 0.3
y_coords = 0.2 * np.cos(frames * 0.05) + frames * 0.01
z_coords = -0.5 * np.sin(frames * 0.1) + 1.0

# Create colormap based on swing phase
colors = np.zeros(len(frames))
phase_boundaries = [0, 15, 30, 45, 60, 75, 90]
for i, boundary in enumerate(phase_boundaries):
    if i < len(phase_boundaries) - 1:
        colors[boundary:phase_boundaries[i+1]] = i

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory with color indicating phase
scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, cmap='viridis', 
                    s=30, alpha=0.8)

# Add annotations for key positions
key_frames = [0, 30, 45, 75]
key_labels = ["Address", "Backswing", "Top", "Impact"]
for i, frame in enumerate(key_frames):
    ax.text(x_coords[frame], y_coords[frame], z_coords[frame], key_labels[i], 
            fontsize=12, color='red')

# Add colorbar with phase labels
cbar = plt.colorbar(scatter, ticks=np.arange(0, 6, 1) + 0.5)
cbar.set_label('Swing Phase')
cbar.set_ticklabels(["Address", "Takeaway", "Backswing", "Top", "Downswing", "Impact"])

# Set labels and title
ax.set_xlabel('X Position (left/right)')
ax.set_ylabel('Y Position (forward/backward)')
ax.set_zlabel('Z Position (up/down)')
ax.set_title('3D Trajectory of Right Wrist During Golf Swing')

# Set consistent axis limits for better proportion
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 1.5)
ax.set_zlim(0, 2)

plt.tight_layout()
plt.savefig('wrist_3d_trajectory.png', dpi=300)
plt.show()