import cv2
import numpy as np
import mediapipe_compat as mp
import os
import math


def process_dual_videos_with_single_3d_skeleton(video1_path, video2_path, output_dir='single_3d_skeleton', fps=30):
    """
    Process two videos with MediaPipe, combine their pose data, and create a single 3D
    skeleton visualization that stands properly on the ground.

    Args:
        video1_path: Path to first camera video
        video2_path: Path to second camera video
        output_dir: Directory to save outputs
        fps: Frame rate for output videos
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Open video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print(f"Error: Could not open one or both video files")
        return

    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use minimum frame count from both videos
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(total_frames1, total_frames2)

    print(f"Processing videos: {video1_path} and {video2_path}")
    print(f"Video 1: {width1}x{height1}, {total_frames1} frames")
    print(f"Video 2: {width2}x{height2}, {total_frames2} frames")
    print(f"Will process a total of {total_frames} frames")

    # Create video writers for output videos
    annotated_video_path = os.path.join(output_dir, 'annotated_dual_video.mp4')
    single_3d_skeleton_path = os.path.join(output_dir, 'combined_3d_pose.mp4')

    # Create writer for annotated side-by-side video
    annotated_width = width1 + width2
    annotated_height = max(height1, height2)
    annotated_writer = cv2.VideoWriter(
        annotated_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (annotated_width, annotated_height)
    )

    # Create writer for 3D skeleton visualization
    # Using a slightly larger size for better visualization
    skeleton_width = 800
    skeleton_height = 800
    skeleton_writer = cv2.VideoWriter(
        single_3d_skeleton_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (skeleton_width, skeleton_height)
    )

    # Define skeleton connections
    skeleton_connections = [
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

    # Function to combine landmarks from two camera views
    def combine_landmarks(landmarks1, landmarks2, confidence1, confidence2):
        """Combine landmarks from two camera views based on detection confidence"""
        if landmarks1 is None and landmarks2 is None:
            return None

        if landmarks1 is None:
            return landmarks2

        if landmarks2 is None:
            return landmarks1

        # Convert to numpy arrays if they aren't already
        lm1 = np.array(landmarks1)
        lm2 = np.array(landmarks2)
        conf1 = np.array(confidence1)
        conf2 = np.array(confidence2)

        # Combine landmarks based on confidence
        # For each keypoint, use the one with higher confidence
        combined = np.zeros_like(lm1)
        for i in range(min(len(lm1), len(lm2))):
            if conf1[i] > conf2[i]:
                combined[i] = lm1[i]
            else:
                combined[i] = lm2[i]

        return combined

    # Function to project 3D points to 2D
    def project_3d_to_2d(points_3d, rotation_x=0, rotation_y=0, rotation_z=0, scale=200, center_x=400, center_y=400):
        """
        Project 3D points to 2D with rotation, scaling, and translation

        Args:
            points_3d: 3D points to project
            rotation_x: Rotation angle around X axis in degrees
            rotation_y: Rotation angle around Y axis in degrees
            rotation_z: Rotation angle around Z axis in degrees
            scale: Scaling factor
            center_x: X center for translation
            center_y: Y center for translation

        Returns:
            2D points after projection
        """
        # Convert rotation angles to radians
        rx = np.radians(rotation_x)
        ry = np.radians(rotation_y)
        rz = np.radians(rotation_z)

        # Create rotation matrices
        # Rotation around X axis
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        # Rotation around Y axis
        rotation_matrix_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        # Rotation around Z axis
        rotation_matrix_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x

        # Apply rotation
        rotated_points = np.array([rotation_matrix @ point for point in points_3d])

        # Project to 2D (simple orthographic projection)
        points_2d = []
        for point in rotated_points:
            x = int(point[0] * scale + center_x)
            y = int(point[1] * scale + center_y)
            points_2d.append((x, y, point[2]))  # Keep z for depth ordering

        return points_2d

    # Function to adjust skeleton to stand on the ground
    def adjust_skeleton_to_ground(landmarks):
        """
        Adjust skeleton so feet touch the ground
        Also applies scaling for better visualization
        """
        # Find the lowest points (usually feet)
        foot_indices = [27, 28, 29, 30, 31, 32]  # Include all foot-related points
        min_y = float('inf')

        for idx in foot_indices:
            if idx < len(landmarks):
                min_y = min(min_y, landmarks[idx, 1])

        # If we found valid foot points, adjust the entire skeleton
        adjusted = landmarks.copy()
        if min_y != float('inf'):
            # Shift the skeleton so feet touch the ground (y=0)
            adjusted[:, 1] -= min_y

            # Scale the skeleton to a reasonable size
            # Measure the height (distance from feet to head)
            head_idx = 0  # Head index
            if head_idx < len(adjusted):
                current_height = adjusted[head_idx, 1]
                if current_height > 0:  # Avoid division by zero
                    # Target height (in skeleton coordinate system)
                    target_height = 1.5
                    scale_factor = target_height / current_height

                    # Scale all coordinates except y-shift
                    adjusted *= scale_factor

        return adjusted

    # Function to draw the ground grid
    def draw_ground_grid(image, points_2d, center_x, center_y, scale=200, grid_size=4, grid_step=0.5):
        """
        Draw a ground grid for better orientation

        Args:
            image: Image to draw on
            points_2d: Projected 2D points (with Z component)
            center_x, center_y: Center of projection
            scale: Scale factor used in projection
            grid_size: Size of the grid (in 3D units)
            grid_step: Step size between grid lines
        """
        # Find the lowest Z value from projected points (for depth ordering)
        min_z = min([p[2] for p in points_2d]) - 0.1

        # Create grid points in 3D
        grid_range = np.arange(-grid_size / 2, grid_size / 2 + grid_step, grid_step)
        grid_points_3d = []

        # Generate grid lines along X and Z axes
        for x in grid_range:
            for z in grid_range:
                grid_points_3d.append([x, 0, z])  # Points along X axis
                grid_points_3d.append([z, 0, x])  # Points along Z axis

        # Project grid points to 2D
        grid_points_2d = project_3d_to_2d(
            grid_points_3d,
            rotation_x=30, rotation_y=0, rotation_z=45,
            scale=scale, center_x=center_x, center_y=center_y
        )

        # Draw grid lines
        drawn_lines = set()
        for i in range(0, len(grid_points_3d), 2):
            if i + 1 < len(grid_points_3d):
                pt1 = (grid_points_2d[i][0], grid_points_2d[i][1])
                pt2 = (grid_points_2d[i + 1][0], grid_points_2d[i + 1][1])

                # Create a unique key for this line to avoid duplicates
                line_key = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]), max(pt1[0], pt2[0]), max(pt1[1], pt2[1]))

                if line_key not in drawn_lines:
                    # Draw light grid lines
                    cv2.line(image, pt1, pt2, (50, 50, 50), 1)
                    drawn_lines.add(line_key)

        # Draw a darker perimeter for the grid
        perimeter_points_3d = [
            [-grid_size / 2, 0, -grid_size / 2],
            [grid_size / 2, 0, -grid_size / 2],
            [grid_size / 2, 0, grid_size / 2],
            [-grid_size / 2, 0, grid_size / 2],
            [-grid_size / 2, 0, -grid_size / 2]
        ]

        perimeter_points_2d = project_3d_to_2d(
            perimeter_points_3d,
            rotation_x=30, rotation_y=0, rotation_z=45,
            scale=scale, center_x=center_x, center_y=center_y
        )

        # Draw perimeter
        for i in range(len(perimeter_points_2d) - 1):
            pt1 = (perimeter_points_2d[i][0], perimeter_points_2d[i][1])
            pt2 = (perimeter_points_2d[i + 1][0], perimeter_points_2d[i + 1][1])
            cv2.line(image, pt1, pt2, (100, 100, 100), 2)

        return image

    # Function to draw the 3D skeleton on a 2D image
    def draw_3d_skeleton(landmarks, width=800, height=800, rotation_x=30, rotation_y=0, rotation_z=45):
        """
        Draw a 3D skeleton on a 2D image using direct projection

        Args:
            landmarks: 3D landmarks to draw
            width, height: Image dimensions
            rotation_x, rotation_y, rotation_z: Rotation angles in degrees

        Returns:
            Image with drawn skeleton
        """
        # Create a black image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Center of the image
        center_x = width // 2
        center_y = height // 2

        # Scale factor for projection
        scale = 200

        # Adjust landmarks to stand on ground
        adjusted_landmarks = adjust_skeleton_to_ground(landmarks)

        # Project 3D points to 2D
        points_2d = project_3d_to_2d(
            adjusted_landmarks,
            rotation_x=rotation_x, rotation_y=rotation_y, rotation_z=rotation_z,
            scale=scale, center_x=center_x, center_y=center_y
        )

        # Draw ground grid
        image = draw_ground_grid(image, points_2d, center_x, center_y, scale)

        # Draw feet shadows (ellipses where feet meet the ground)
        foot_indices = [27, 28, 31, 32]  # Feet and ankles
        for idx in foot_indices:
            if idx < len(points_2d):
                shadow_x = points_2d[idx][0]
                shadow_y = points_2d[idx][1]
                # Draw foot shadow as an ellipse
                cv2.ellipse(image, (shadow_x, shadow_y), (20, 10), 0, 0, 360, (0, 50, 0), -1)

        # Draw skeleton connections
        for connection in skeleton_connections:
            start_idx, end_idx = connection
            if start_idx < len(points_2d) and end_idx < len(points_2d):
                pt1 = (points_2d[start_idx][0], points_2d[start_idx][1])
                pt2 = (points_2d[end_idx][0], points_2d[end_idx][1])

                # Determine line thickness based on connection type
                if (start_idx in [11, 12, 23, 24] and end_idx in [11, 12, 23, 24]):
                    # Thicker lines for torso
                    thickness = 4
                else:
                    thickness = 3

                # Determine color based on body part
                if start_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    # Head and face
                    color = (0, 255, 255)  # Yellow
                elif (start_idx in [11, 13, 15, 17, 19, 21] or end_idx in [11, 13, 15, 17, 19, 21]):
                    # Left side
                    color = (255, 0, 0)  # Blue
                elif (start_idx in [12, 14, 16, 18, 20, 22] or end_idx in [12, 14, 16, 18, 20, 22]):
                    # Right side
                    color = (0, 255, 0)  # Green
                else:
                    # Other parts
                    color = (255, 255, 255)  # White

                # Draw the connection
                cv2.line(image, pt1, pt2, color, thickness)

        # Draw keypoints
        for i, point in enumerate(points_2d):
            # Determine point size and color based on joint type
            if i in [0, 11, 12, 23, 24]:  # Head, shoulders, hips
                radius = 6
                color = (255, 200, 0)  # Bright yellow
            elif i in [13, 14, 15, 16]:  # Elbows and wrists
                radius = 5
                color = (0, 200, 255)  # Light blue
            elif i in [25, 26, 27, 28]:  # Knees and ankles
                radius = 5
                color = (255, 0, 255)  # Magenta
            else:
                radius = 4
                color = (150, 150, 150)  # Gray

            # Draw the keypoint
            cv2.circle(image, (point[0], point[1]), radius, color, -1)

            # Add small white outline for better visibility
            cv2.circle(image, (point[0], point[1]), radius + 1, (255, 255, 255), 1)

        # Add title and frame info
        return image

    # Create MediaPipe pose instances for each video
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose1, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose2:

        frame_idx = 0
        while frame_idx < total_frames:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # Convert frames to RGB
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # Process frames with MediaPipe
            results1 = pose1.process(frame1_rgb)
            results2 = pose2.process(frame2_rgb)

            # Initialize variables for landmarks and confidence
            landmarks1 = None
            landmarks2 = None
            confidence1 = np.zeros(33)  # MediaPipe has 33 landmarks
            confidence2 = np.zeros(33)

            # Extract landmarks and confidence from results
            if results1.pose_world_landmarks:
                landmarks1 = []
                for i, landmark in enumerate(results1.pose_world_landmarks.landmark):
                    landmarks1.append([landmark.x, landmark.y, landmark.z])
                    confidence1[i] = landmark.visibility
                landmarks1 = np.array(landmarks1)

            if results2.pose_world_landmarks:
                landmarks2 = []
                for i, landmark in enumerate(results2.pose_world_landmarks.landmark):
                    landmarks2.append([landmark.x, landmark.y, landmark.z])
                    confidence2[i] = landmark.visibility
                landmarks2 = np.array(landmarks2)

            # Combine landmarks from both cameras
            combined_landmarks = combine_landmarks(landmarks1, landmarks2, confidence1, confidence2)

            # Create annotated frames
            annotated_frame1 = frame1.copy()
            annotated_frame2 = frame2.copy()

            if results1.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame1,
                    results1.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                # Add camera label
                cv2.putText(annotated_frame1, "Camera 1", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if results2.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame2,
                    results2.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                # Add camera label
                cv2.putText(annotated_frame2, "Camera 2", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Combine annotated frames side by side
            if height1 == height2:
                side_by_side = np.hstack((annotated_frame1, annotated_frame2))
            else:
                # Resize to same height if needed
                h = min(height1, height2)
                w1 = int(width1 * h / height1)
                w2 = int(width2 * h / height2)
                resized1 = cv2.resize(annotated_frame1, (w1, h))
                resized2 = cv2.resize(annotated_frame2, (w2, h))
                side_by_side = np.hstack((resized1, resized2))

            # Add frame info
            cv2.putText(side_by_side, f"Frame: {frame_idx}/{total_frames}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write annotated frame to video
            annotated_writer.write(side_by_side)

            # Create 3D skeleton visualization if we have combined landmarks
            if combined_landmarks is not None:
                # Draw 3D skeleton
                skeleton_img = draw_3d_skeleton(combined_landmarks)

                # Add title and frame counter
                cv2.putText(skeleton_img, "Combined 3D Skeleton",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(skeleton_img, f"Frame: {frame_idx}/{total_frames}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Save frame to image file
                cv2.imwrite(os.path.join(output_dir, f'frame_{frame_idx:04d}.png'), skeleton_img)

                # Write to skeleton video
                skeleton_writer.write(skeleton_img)

            frame_idx += 1

            # Display progress
            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

    # Release resources
    cap1.release()
    cap2.release()
    annotated_writer.release()
    skeleton_writer.release()

    print(f"Processing complete! Output files:")
    print(f"1. Annotated dual video: {annotated_video_path}")
    print(f"2. Combined 3D skeleton video: {single_3d_skeleton_path}")
    print(f"3. Individual frames saved in: {output_dir}")


if __name__ == "__main__":
    # Replace with your actual video paths
    video1_path = "./../PoseVideos/17.mp4"
    video2_path = "./../PoseVideos/18.mp4"

    process_dual_videos_with_single_3d_skeleton(video1_path, video2_path)
