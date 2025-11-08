"""
3D Pose Visualization with MediaPipe and Matplotlib

This script:
1. Loads a video file and processes it with MediaPipe Pose
2. Extracts 3D landmarks from each frame
3. Visualizes the 3D skeleton using matplotlib with proper floor orientation
4. Can either show real-time animation or save to a video file
"""

import os
import cv2
import time
import numpy as np
import mediapipe_compat as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D


class Pose3DVisualizer:
    def __init__(self):
        """Initialize the MediaPipe pose detector and matplotlib settings"""
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Using highest quality model
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Setup for 3D plotting
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Store landmarks for each frame
        self.all_landmarks = []

        # Framerate management
        self.fps = 30
        self.frame_time = 1/self.fps

    def process_video(self, video_path):
        """Process video file and extract 3D landmarks from each frame"""
        # Check if video file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video with {total_frames} frames at {self.fps} FPS...")

        # Process each frame
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose.process(image_rgb)

            # Store 3D landmarks if detected
            if results.pose_world_landmarks:
                # Extract 3D coordinates from MediaPipe results
                landmarks_3d = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark
                ])

                # Process landmarks to reorient them properly
                landmarks_3d = self._reorient_landmarks(landmarks_3d)

                self.all_landmarks.append(landmarks_3d)
            else:
                # If no landmarks detected, use the previous frame's landmarks or empty array
                if self.all_landmarks:
                    self.all_landmarks.append(self.all_landmarks[-1])
                else:
                    self.all_landmarks.append(np.zeros((33, 3)))  # 33 landmarks with x,y,z=0

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        # Release resources
        cap.release()
        print(f"Video processing complete! Extracted landmarks from {len(self.all_landmarks)} frames.")

    def _reorient_landmarks(self, landmarks):
        """
        Reorient the landmarks so the person is standing properly on the X-Z plane

        In MediaPipe world coordinates:
        - X: left/right
        - Y: up/down
        - Z: forward/backward

        We want to transform so:
        - X: left/right (unchanged)
        - Y: up/down (height - unchanged)
        - Z: forward/backward (unchanged)

        But we need to ensure feet are on the ground plane (X-Z plane)
        """
        # Make a copy to avoid modifying the original
        reoriented = landmarks.copy()

        # Find the lowest foot point (highest y-value in MediaPipe coordinates) to use as the floor
        # Feet landmarks are indices 27-32 (left ankle, left heel, left foot index, right ankle, right heel, right foot index)
        foot_indices = [27, 28, 29, 30, 31, 32]
        foot_y_values = [landmarks[i, 1] for i in foot_indices if i < len(landmarks)]

        if foot_y_values:
            # Find the lowest point of the feet (this will be our ground reference)
            ground_level = max(foot_y_values)

            # Adjust all landmarks so the lowest foot point is at y=0
            y_offset = ground_level
            reoriented[:, 1] -= y_offset

            # Flip Y axis for more intuitive visualization (make Y go up instead of down)
            reoriented[:, 1] = -reoriented[:, 1]

        return reoriented

    def _get_pose_connections(self):
        """Get the joint connections used by MediaPipe Pose"""
        return [(connection[0], connection[1]) for connection in self.mp_pose.POSE_CONNECTIONS]

    def _update_plot(self, frame_idx):
        """Update the 3D plot for a specific frame"""
        self.ax.clear()

        # Get landmarks for current frame
        if frame_idx >= len(self.all_landmarks):
            return

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
        self.ax.plot_surface(XX, ZZ, YY, alpha=0.2, color='gray')

        # Plot all landmarks
        self.ax.scatter(
            landmarks[:, 0],  # X
            landmarks[:, 2],  # Z
            landmarks[:, 1],  # Y
            color='blue', s=20
        )

        # Draw connections between joints
        connections = self._get_pose_connections()
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                xs = [landmarks[start_idx, 0], landmarks[end_idx, 0]]
                zs = [landmarks[start_idx, 2], landmarks[end_idx, 2]]
                ys = [landmarks[start_idx, 1], landmarks[end_idx, 1]]
                self.ax.plot(xs, zs, ys, color='red', linewidth=2)

        # Set appropriate axis limits
        max_range = 2.0
        self.ax.set_xlim(-max_range/2, max_range/2)  # X
        self.ax.set_ylim(-max_range/2, max_range/2)  # Z
        self.ax.set_zlim(0, max_range)               # Y - start from 0 for the floor

        # Adjust the viewing angle
        self.ax.view_init(elev=15, azim=(frame_idx % 360))

        return self.ax

    def animate(self, show=True, save=False, output_path='output_3d_skeleton.mp4'):
        """
        Create an animation of the 3D pose.

        Args:
            show (bool): Whether to display the animation in real-time
            save (bool): Whether to save the animation to a video file
            output_path (str): Path to save the output video if save=True
        """
        if len(self.all_landmarks) == 0:
            raise ValueError("No landmarks to animate. Process a video first.")

        num_frames = len(self.all_landmarks)
        print(f"Creating animation with {num_frames} frames...")

        # Create animation
        ani = FuncAnimation(
            self.fig, 
            self._update_plot, 
            frames=num_frames,
            interval=self.frame_time * 1000,  # Convert to milliseconds
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
                metadata=dict(artist='MediaPipe 3D Pose'),
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


def main():
    """Main function to run the 3D pose visualization"""
    # Set up the visualizer
    visualizer = Pose3DVisualizer()

    # Process a video file
    video_path = "PoseVideos/18.mp4"  # Change this to your video file path

    # Check if the video file exists, if not, suggest looking for alternatives
    if not os.path.isfile(video_path):
        print(f"Video file '{video_path}' not found.")
        # Check if PoseVideos directory exists and has any mp4 files
        if os.path.exists("PoseVideos"):
            video_files = [f for f in os.listdir("PoseVideos") if f.endswith(".mp4")]
            if video_files:
                video_path = os.path.join("PoseVideos", video_files[0])
                print(f"Using alternative video file: {video_path}")
            else:
                print("No .mp4 files found in PoseVideos directory.")
                return
        else:
            print("Please provide a valid video file path.")
            return

    # Create output_videos directory if it doesn't exist
    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Set output path in the output_videos folder
    output_path = os.path.join(output_dir, "output_3d_skeleton.mp4")

    try:
        # Process video and extract 3D landmarks
        visualizer.process_video(video_path)

        # Create and show/save the animation
        # Toggle these as needed:
        show_animation = True   # Set to True to display animation in real-time
        save_animation = True   # Set to True to save animation as video file

        visualizer.animate(
            show=show_animation,
            save=save_animation,
            output_path=output_path
        )

        print(f"Final 3D skeleton video saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
