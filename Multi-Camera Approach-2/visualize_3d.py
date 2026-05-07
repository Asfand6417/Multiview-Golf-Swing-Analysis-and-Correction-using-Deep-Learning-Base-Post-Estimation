"""
3D Skeleton Visualization with Matplotlib/Plotly

This module creates 3D animations of golf swing skeletons with proper
visual specifications and ground plane alignment.

Research Objectives:
    1. Accuracy of 3D reconstruction [[45], [46]]
    2. Comparison with single-view & marker-based baselines [[50]-[52]]
    
Technical Constraints:
    - 3D axes: x = left→right, y = up, z = forward (toward ball)
    - Feet should be anchored to z=0 ground plane
    - Use a single line colour (#00AEEF) for limbs and red key-points (#FF0000)
    - Remove black background; use transparent/white background
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


# Hyperparameters for visualization
LIMB_COLOR = "#00AEEF"  # Specified blue color for limbs
KEYPOINT_COLOR = "#FF0000"  # Red color for keypoints
KEYPOINT_SIZE = 6  # Size of keypoints
LIMB_WIDTH = 3  # Width of limbs
FPS = 30  # Default frames per second
DPI = 100  # Resolution for output video
FIGURE_SIZE = (10, 10)  # Size of the figure in inches
ELEVATION = 15  # Default camera elevation angle
AZIMUTH_START = 45  # Starting azimuth angle
ROTATE_VIEW = True  # Whether to rotate the view during animation


class Skeleton3DVisualizer:
    """
    Class for visualizing 3D skeletons with proper visual specifications.
    
    Research Alignment:
        Supports Objective 1: Accuracy of 3D reconstruction [[45], [46]]
        by providing clear visualization of the reconstructed skeleton.
        
        Supports Objective 4: Comparison with baselines [[50]-[52]]
        by enabling visual comparison of different reconstruction methods.
    """
    
    def __init__(self, bg_color='white'):
        """
        Initialize the 3D skeleton visualizer.
        
        Args:
            bg_color: Background color ('white', 'transparent', or any valid color)
        """
        # Setup for 3D plotting
        self.fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
        
        # Set background color
        if bg_color == 'transparent':
            self.ax = self.fig.add_subplot(111, projection='3d', facecolor='none')
            self.fig.patch.set_alpha(0.0)
        else:
            self.ax = self.fig.add_subplot(111, projection='3d', facecolor=bg_color)
            self.fig.patch.set_facecolor(bg_color)
        
        # Store landmarks for each frame
        self.all_landmarks = []
        
        # Framerate management
        self.fps = FPS
        
        # Define skeleton connections (MediaPipe pose connections)
        self.connections = [
            # Torso
            (11, 12), (12, 24), (24, 23), (23, 11),
            # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Hands
            (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
            # Legs
            (23, 25), (25, 27), (24, 26), (26, 28),
            # Feet
            (27, 29), (27, 31), (28, 30), (28, 32),
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10)
        ]
    
    def load_landmarks(self, landmarks_sequence):
        """
        Load a sequence of 3D landmarks.
        
        Args:
            landmarks_sequence: List of arrays, each containing 3D landmarks for one frame
        """
        self.all_landmarks = landmarks_sequence
        print(f"Loaded {len(self.all_landmarks)} frames of 3D landmarks")
    
    def _update_plot(self, frame_idx):
        """
        Update the 3D plot for a specific frame.
        
        Args:
            frame_idx: Index of the frame to visualize
            
        Returns:
            The updated axis object
        """
        self.ax.clear()
        
        # Get landmarks for current frame
        if frame_idx >= len(self.all_landmarks):
            return self.ax
            
        landmarks = self.all_landmarks[frame_idx]
        
        # Set axis labels and title
        self.ax.set_xlabel('X (left/right)')
        self.ax.set_ylabel('Z (forward/backward)')
        self.ax.set_zlabel('Y (up/down)')
        self.ax.set_title(f'3D Pose - Frame {frame_idx}')
        
        # Draw a grid on the X-Z plane (floor)
        x_min, x_max = -1, 1
        z_min, z_max = -1, 1
        XX, ZZ = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
        YY = np.zeros_like(XX)
        self.ax.plot_surface(XX, ZZ, YY, alpha=0.2, color='lightgray')
        
        # Plot all landmarks with red color
        self.ax.scatter(
            landmarks[:, 0],  # X
            landmarks[:, 2],  # Z
            landmarks[:, 1],  # Y
            color=KEYPOINT_COLOR, s=KEYPOINT_SIZE**2
        )
        
        # Draw connections between joints with specified blue color
        for connection in self.connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                xs = [landmarks[start_idx, 0], landmarks[end_idx, 0]]
                zs = [landmarks[start_idx, 2], landmarks[end_idx, 2]]
                ys = [landmarks[start_idx, 1], landmarks[end_idx, 1]]
                self.ax.plot(xs, zs, ys, color=LIMB_COLOR, linewidth=LIMB_WIDTH)
        
        # Set appropriate axis limits
        max_range = 2.0
        self.ax.set_xlim(-max_range/2, max_range/2)  # X
        self.ax.set_ylim(-max_range/2, max_range/2)  # Z
        self.ax.set_zlim(0, max_range)               # Y - start from 0 for the floor
        
        # Adjust the viewing angle
        if ROTATE_VIEW:
            self.ax.view_init(elev=ELEVATION, azim=(AZIMUTH_START + frame_idx % 360))
        else:
            self.ax.view_init(elev=ELEVATION, azim=AZIMUTH_START)
        
        return self.ax
    
    def animate(self, show=True, save=False, output_path='output_3d_skeleton.mp4'):
        """
        Create an animation of the 3D pose.
        
        Args:
            show: Whether to display the animation in real-time
            save: Whether to save the animation to a video file
            output_path: Path to save the output video if save=True
            
        Returns:
            Animation object
        """
        if len(self.all_landmarks) == 0:
            raise ValueError("No landmarks to animate. Load landmarks first.")
            
        num_frames = len(self.all_landmarks)
        print(f"Creating animation with {num_frames} frames...")
        
        # Create animation
        ani = FuncAnimation(
            self.fig, 
            self._update_plot, 
            frames=num_frames,
            interval=1000/self.fps,  # Convert fps to interval in milliseconds
            blit=False
        )
        
        # Save animation if requested
        if save:
            print(f"Saving animation to {output_path}...")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Setup FFmpeg writer
            writer = FFMpegWriter(
                fps=self.fps, 
                metadata=dict(artist='Golf Pose 3D'),
                bitrate=2000
            )
            
            # Save the animation
            try:
                ani.save(output_path, writer=writer)
                print(f"Animation saved to {output_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Make sure FFmpeg is installed and in your PATH.")
        
        # Show the animation if requested
        if show:
            plt.tight_layout()
            plt.show()
            
        return ani


def visualize_sequence(landmarks_sequence, output_path=None, show=True, bg_color='white'):
    """
    Visualize a sequence of 3D landmarks.
    
    Args:
        landmarks_sequence: List of arrays, each containing 3D landmarks for one frame
        output_path: Path to save the output video (if None, won't save)
        show: Whether to display the animation in real-time
        bg_color: Background color ('white', 'transparent', or any valid color)
        
    Returns:
        Animation object
    """
    visualizer = Skeleton3DVisualizer(bg_color=bg_color)
    visualizer.load_landmarks(landmarks_sequence)
    
    return visualizer.animate(
        show=show,
        save=output_path is not None,
        output_path=output_path if output_path else 'output_3d_skeleton.mp4'
    )


if __name__ == "__main__":
    # Example usage
    print("3D Visualization module - Run through the main pipeline")