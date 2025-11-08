import os
import cv2
import time  # Not strictly used in the final version, but often useful
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import traceback


class Pose3DVisualizer:
    def __init__(self):
        """Initialize the MediaPipe pose detector and matplotlib settings"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Changed to 1 for broader compatibility/robustness
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.all_landmarks = []

        self.fps = 30  # Default FPS
        self.frame_time = 1 / self.fps

    def reset_figure(self):
        """Clears and re-initializes the figure and axes for a new animation."""
        self.fig.clf()  # Clear the current figure
        self.ax = self.fig.add_subplot(111, projection='3d')  # Re-add subplot

    def process_video(self, video_path):
        """Process video file and extract 3D landmarks from each frame"""
        self.all_landmarks = []  # Clear previous landmarks for a new video

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            print("Warning: Video FPS reported as 0. Using default 30 FPS.")
            self.fps = 30
        self.frame_time = 1 / self.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video '{os.path.basename(video_path)}': {total_frames} frames at {self.fps:.2f} FPS...")

        processed_frames_with_landmarks = 0
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                if frame_count < total_frames and total_frames > 0:  # Check if it's an unexpected early end
                    print(f"Warning: Video ended prematurely or failed to read frame {frame_count + 1}/{total_frames}.")
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_world_landmarks:
                processed_frames_with_landmarks += 1
                landmarks_3d = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark
                ])

                if processed_frames_with_landmarks == 1 and frame_count < 5:  # Print for one of the first few frames
                    print(
                        f"  First detection: Frame {frame_count}, Nose (idx 0) original world coords: X={landmarks_3d[0, 0]:.2f}, Y={landmarks_3d[0, 1]:.2f}, Z={landmarks_3d[0, 2]:.2f}")

                landmarks_3d = self._reorient_landmarks(landmarks_3d)
                self.all_landmarks.append(landmarks_3d)
            else:
                if self.all_landmarks:
                    self.all_landmarks.append(self.all_landmarks[-1])
                else:
                    # Number of landmarks for pose is 33 (0-32)
                    self.all_landmarks.append(np.zeros((33, 3)))

            frame_count += 1
            # Print progress update, e.g., every 10% or every 50 frames
            if total_frames > 0 and frame_count % (max(1, total_frames // 20)) == 0:
                print(
                    f"  Processed {frame_count}/{total_frames} frames. Landmarks found in {processed_frames_with_landmarks} frames so far.")
            elif frame_count % 50 == 0:  # Fallback for very long/unknown length videos
                print(
                    f"  Processed {frame_count} frames. Landmarks found in {processed_frames_with_landmarks} frames so far.")

        cap.release()
        print(f"Video processing complete! Extracted landmarks for {len(self.all_landmarks)} frames.")
        if processed_frames_with_landmarks == 0 and frame_count > 0:
            print("WARNING: No landmarks were detected in any frame of the video.")
        elif frame_count > 0:
            print(
                f"Landmarks were successfully detected in {processed_frames_with_landmarks}/{frame_count} processed frames.")

    def _reorient_landmarks(self, landmarks):
        reoriented = landmarks.copy()

        foot_indices = [self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
                        self.mp_pose.PoseLandmark.LEFT_HEEL.value,
                        self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
                        self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                        self.mp_pose.PoseLandmark.RIGHT_HEEL.value,
                        self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        foot_y_values = [landmarks[i, 1] for i in foot_indices if i < len(landmarks)]

        if foot_y_values:
            ground_level = max(foot_y_values)  # MediaPipe Y is inverted (smaller is higher)
            y_offset = ground_level
            reoriented[:, 1] -= y_offset
            reoriented[:, 1] = -reoriented[:, 1]  # Flip Y to make positive Y "up"

        return reoriented

    def _get_pose_connections(self):
        return list(self.mp_pose.POSE_CONNECTIONS)  # Ensure it's a list of tuples

    def _update_plot(self, frame_idx):
        self.ax.clear()

        if frame_idx >= len(self.all_landmarks):
            print(f"Warning: Frame index {frame_idx} out of bounds for landmarks list (len {len(self.all_landmarks)}).")
            return self.ax  # Return axis to prevent error with blit=True if it were used

        landmarks = self.all_landmarks[frame_idx]

        # Axis labels reflect MediaPipe world coordinate conventions AFTER reorientation for Y
        self.ax.set_xlabel('X (left/right)')
        self.ax.set_ylabel('Z (forward/backward)')  # Matplotlib's Y-axis on the plot
        self.ax.set_zlabel('Y (up/down)')  # Matplotlib's Z-axis on the plot (height)
        self.ax.set_title(f'3D Pose - Frame {frame_idx}')

        # Floor grid on the X-Z plane (at Y=0 after reorientation)
        x_min, x_max = -1.5, 1.5  # Expanded slightly for better visualization
        z_min, z_max = -1.5, 1.5
        XX, ZZ = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
        YY = np.zeros_like(XX)  # Floor is at Y=0 height
        # Plotting floor on Matplotlib's X-Y plane, assuming Z is height.
        # Given our scatter: X_body -> MPL_X, Z_body -> MPL_Y, Y_body -> MPL_Z
        # So floor should be on MPL_X, MPL_Y plane.
        self.ax.plot_surface(XX, ZZ, YY, alpha=0.2, color='gray')

        # Scatter plot:
        # landmarks[:, 0] is X_body
        # landmarks[:, 2] is Z_body
        # landmarks[:, 1] is Y_body (height, reoriented)
        self.ax.scatter(
            landmarks[:, 0],  # Plot X_body on Matplotlib X-axis
            landmarks[:, 2],  # Plot Z_body on Matplotlib Y-axis
            landmarks[:, 1],  # Plot Y_body (height) on Matplotlib Z-axis
            color='blue', s=20
        )

        connections = self._get_pose_connections()
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                xs = [landmarks[start_idx, 0], landmarks[end_idx, 0]]  # X_body
                zs = [landmarks[start_idx, 2], landmarks[end_idx, 2]]  # Z_body
                ys = [landmarks[start_idx, 1], landmarks[end_idx, 1]]  # Y_body (height)
                self.ax.plot(xs, zs, ys, color='red', linewidth=2)  # Plots on MPL X, Y, Z axes

        max_range_body = 2.0  # Max expected range for a person's height or width in meters
        self.ax.set_xlim(-max_range_body / 1.5, max_range_body / 1.5)  # X_body limits
        self.ax.set_ylim(-max_range_body / 1.5, max_range_body / 1.5)  # Z_body limits
        self.ax.set_zlim(0, max_range_body)  # Y_body (height) limits, floor at 0

        # Consistent viewing angle or rotating:
        # For the image provided, azim appears to be around 270-300 to make Z (body) horizontal.
        # elev=15, azim = (frame_idx * 2 % 360) # Slower rotation
        self.ax.view_init(elev=20, azim=-60 + (frame_idx % 360))  # Start from a common side view, then rotate
        # For static view matching image (approx): self.ax.view_init(elev=15, azim=280)

        return self.ax  # For blit=True (though blit=False is used here)

    def animate(self, show=True, save=False, output_path='output_3d_skeleton.mp4'):
        if not self.all_landmarks:
            print("Error: No landmarks to animate. Process a video first.")
            return None  # Return None if no animation can be created

        num_frames = len(self.all_landmarks)
        if num_frames == 0:
            print("Error: No frames with landmarks available for animation.")
            return None

        print(f"Creating animation with {num_frames} frames...")

        # Ensure figure is clean before starting animation if reusing visualizer instance
        # self.reset_figure() # Call this if fig/ax might be stale from previous use.

        ani = FuncAnimation(
            self.fig,
            self._update_plot,
            frames=num_frames,
            interval=max(1, int(self.frame_time * 1000)),  # Convert to ms, ensure positive
            blit=False  # Blit=True can be faster but sometimes problematic with 3D
        )

        if save:
            print(f"Attempting to save animation to {output_path}...")
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            writer = FFMpegWriter(
                fps=self.fps,
                metadata=dict(artist='MediaPipe 3D Pose Visualizer'),
                bitrate=2000  # Increased bitrate for better quality
            )
            try:
                ani.save(output_path, writer=writer)
                print(f"Animation successfully saved to {output_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Please ensure FFmpeg is installed and accessible in your system's PATH.")
                print("You can download FFmpeg from https://ffmpeg.org/download.html")

        if show:
            print("Displaying animation...")
            plt.tight_layout()
            try:
                plt.show()
            except Exception as e:
                print(f"Error displaying animation: {e}")

        return ani


def main():
    visualizer = Pose3DVisualizer()

    default_video_dir = "PoseVideos"
    default_video_file = "6.mp4"  # Default specific file to try

    # Construct full default path
    full_default_path = os.path.join(default_video_dir, default_video_file)

    user_video_path = input(f"Enter video path (e.g., {default_video_dir}/myvideo.mp4)\n"
                            f"Press Enter for default ({full_default_path}): ").strip()

    video_path_to_process = user_video_path if user_video_path else full_default_path

    if not os.path.isfile(video_path_to_process):
        print(f"ERROR: Specified video file '{video_path_to_process}' not found.")

        if os.path.exists(default_video_dir):
            print(f"\nLooking for .mp4 files in '{default_video_dir}' directory...")
            available_videos = [f for f in os.listdir(default_video_dir) if
                                f.lower().endswith((".mp4", ".avi", ".mov", ".wmv"))]

            if available_videos:
                print("Available video files:")
                for i, fname in enumerate(available_videos):
                    print(f"  {i + 1}. {fname}")

                try:
                    choice_str = input(
                        f"Enter number of video to use (1-{len(available_videos)}), or 0 to exit: ").strip()
                    if not choice_str:  # User pressed Enter
                        print("No selection made. Exiting.")
                        return

                    choice_idx = int(choice_str) - 1
                    if 0 <= choice_idx < len(available_videos):
                        video_path_to_process = os.path.join(default_video_dir, available_videos[choice_idx])
                        print(f"Selected video: {video_path_to_process}")
                    elif int(choice_str) == 0:
                        print("Exiting.")
                        return
                    else:
                        print("Invalid choice. Exiting.")
                        return
                except ValueError:
                    print("Invalid input (not a number). Exiting.")
                    return
                except Exception as e:
                    print(f"An error occurred during selection: {e}. Exiting.")
                    return
            else:
                print(f"No suitable video files found in '{default_video_dir}'. Please ensure videos are present.")
                return
        else:
            print(f"Default video directory '{default_video_dir}' not found. Please check the path.")
            return

    if not os.path.isfile(video_path_to_process):  # Final check
        print(f"ERROR: Video file '{video_path_to_process}' still not resolved or found. Exiting.")
        return

    print(f"\nAttempting to process video: {video_path_to_process}")

    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    base_video_name = os.path.splitext(os.path.basename(video_path_to_process))[0]
    output_filename = f"output_3d_{base_video_name}.mp4"
    final_output_path = os.path.join(output_dir, output_filename)

    try:
        visualizer.process_video(video_path_to_process)

        if not visualizer.all_landmarks or all(np.all(lm == 0) for lm in visualizer.all_landmarks):
            print("\nWarning: No valid landmarks were extracted. Animation might be empty or static.")
        elif len(visualizer.all_landmarks) < 2:
            print("\nWarning: Only one frame of landmarks extracted. Animation will be static (a single pose).")

        show_animation = True
        save_animation = True

        # Before creating a new animation, ensure the figure is clean, especially if main() could be looped
        visualizer.reset_figure()

        animation_object = visualizer.animate(
            show=show_animation,
            save=save_animation,
            output_path=final_output_path
        )

        if not animation_object and save_animation:
            print(f"Animation could not be created or saved to {final_output_path}.")
        elif save_animation:  # Implies animation_object is not None
            pass  # Save message is in animate()

        if not show_animation and not save_animation:
            print("Processing complete. Animation was not shown or saved as per settings.")

    except FileNotFoundError as e:
        print(f"File Error: {e}")
    except ValueError as e:
        print(f"Value Error during processing or animation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()