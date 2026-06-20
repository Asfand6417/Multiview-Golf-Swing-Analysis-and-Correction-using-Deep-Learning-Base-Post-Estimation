import os
import numpy as np

import cv2

from multicamera.Complete_3D_Visualization_System import create_3d_video
from multicamera.Joint_Position_Optimization_and_Confidence_Based_Integration import optimize_keypoints
from multicamera.camera_calibration import calibrate_cameras
from multicamera.camera_setup import setup_cameras
from multicamera.multiview_poseestimation import extract_keypoints


def triangulate_points(keypoints1_opt, keypoints2_opt, calibration_data):
    """Triangulate 3D points from corresponding 2D points in both camera views"""
    # Get calibration parameters
    mtx1 = calibration_data['camera1_matrix']
    mtx2 = calibration_data['camera2_matrix']
    R = calibration_data['R']
    T = calibration_data['T']

    # Compute projection matrices for both cameras
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # [I|0]
    P1 = mtx1 @ P1

    # Compute projection matrix for camera 2
    RT = np.hstack((R, T))
    P2 = mtx2 @ RT

    # Initialize array for 3D points
    points_3d = []

    for kp1, kp2 in zip(keypoints1_opt, keypoints2_opt):
        # Convert points to correct format
        point1 = np.array([[kp1[0]], [kp1[1]]], dtype=np.float32)
        point2 = np.array([[kp2[0]], [kp2[1]]], dtype=np.float32)

        # Triangulate one point
        point_4d = cv2.triangulatePoints(P1, P2, point1, point2)

        # Convert to 3D homogeneous coordinates
        point_3d = point_4d[:3] / point_4d[3]

        points_3d.append(point_3d.flatten())

    return np.array(points_3d)


def main():
    """Main execution function for dual-camera 3D reconstruction using video files"""
    # Setup directory for 3D reconstruction frames
    os.makedirs('3d_reconstruction', exist_ok=True)

    # Step 1: Calibrate cameras (keep your existing calibration code)
    if not os.path.exists('calibration_data.npy'):
        print("Performing camera calibration...")
        calibration_data = calibrate_cameras()
    else:
        print("Loading existing camera calibration...")
        calibration_data = np.load('calibration_data.npy', allow_pickle=True).item()

    # Step 2: Use video files instead of live cameras
    video1_path = "./../PoseVideos/17.mp4"  # Change to your first video file path
    video2_path = "./../PoseVideos/18.mp4"  # Change to your second video file path

    print(f"Using video files:\n1: {video1_path}\n2: {video2_path}")

    cam1 = cv2.VideoCapture(video1_path)
    cam2 = cv2.VideoCapture(video2_path)

    if not cam1.isOpened():
        print(f"Error: Could not open video file: {video1_path}")
        return

    if not cam2.isOpened():
        print(f"Error: Could not open video file: {video2_path}")
        return

    # Get initial frames to determine video dimensions
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("Failed to read initial frames from video files")
        return

    # Set up video writers for output videos
    height1, width1 = frame1.shape[:2]
    height2, width2 = frame2.shape[:2]

    # Create output directory for processed videos
    output_dir = '3d_reconstruction'

    # Define paths for output videos
    output_video1_path = os.path.join(output_dir, 'processed_video1.mp4')
    output_video2_path = os.path.join(output_dir, 'processed_video2.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get frame rates of videos
    fps1 = cam1.get(cv2.CAP_PROP_FPS)
    fps2 = cam2.get(cv2.CAP_PROP_FPS)

    # Use the minimum fps for the output videos
    output_fps = min(fps1, fps2)

    # Update the VideoWriter fps
    cam1_writer = cv2.VideoWriter(output_video1_path, fourcc, output_fps, (width1, height1))
    cam2_writer = cv2.VideoWriter(output_video2_path, fourcc, output_fps, (width2, height2))

    # The rest of your code remains the same...

    # Define MediaPipe joint connections for visualization
    # These are the indices of connected joints in MediaPipe pose
    joint_connections = [
        # Torso
        (11, 12), (12, 24), (24, 23), (23, 11),
        # Left arm
        (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
        # Right arm
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
        # Left leg
        (23, 25), (25, 27), (27, 29), (29, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (30, 32)
    ]

    # Step 3: Process synchronized frames
    frame_num = 0
    prev_points_3d = None  # Initialize for first frame

    try:
        # Use the initial frames we already read
        while True:
            if frame_num > 0:  # Skip for first iteration since we already read frames
                ret1, frame1 = cam1.read()
                ret2, frame2 = cam2.read()

                if not ret1 or not ret2:
                    break

            # Extract keypoints from both views
            keypoints1, conf1, frame1_annotated = extract_keypoints(frame1)
            keypoints2, conf2, frame2_annotated = extract_keypoints(frame2)

            # Write the annotated frames to the video files
            cam1_writer.write(frame1_annotated)
            cam2_writer.write(frame2_annotated)

            if len(keypoints1) == 0 or len(keypoints2) == 0:
                frame_num += 1
                continue

            # Optimize keypoints using confidence scores to handle occlusions
            keypoints1_opt, keypoints2_opt, conf_combined = optimize_keypoints(
                keypoints1, conf1, keypoints2, conf2, calibration_data
            )

            # Triangulate 3D points
            points_3d = triangulate_points(keypoints1_opt, keypoints2_opt, calibration_data)

            # Apply anatomical constraints to ensure valid skeleton
            points_3d = apply_anatomical_constraints(points_3d)

            # Smooth 3D trajectories
            if frame_num > 0 and prev_points_3d is not None:
                points_3d = smooth_trajectories(points_3d, prev_points_3d)

            # Visualize 3D reconstruction
            visualize_3d_skeleton(points_3d, joint_connections, frame_num)

            # Display annotated 2D views
            cv2.imshow('Camera 1 View', frame1_annotated)
            cv2.imshow('Camera 2 View', frame2_annotated)

            # Store current points for next frame smoothing
            prev_points_3d = points_3d.copy()

            # Increment frame counter
            frame_num += 1

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up resources
        cam1.release()
        cam2.release()

        # Release video writers
        cam1_writer.release()
        cam2_writer.release()

        cv2.destroyAllWindows()

        # Create output_videos directory if it doesn't exist
        output_videos_dir = 'output_videos'
        os.makedirs(output_videos_dir, exist_ok=True)

        # Create 3D visualization video
        output_3d_video = os.path.join(output_videos_dir, 'output_3d_skeleton.mp4')
        create_3d_video(output_3d_video)

        print(f"Processed {frame_num} frames successfully")
        print(f"Camera 1 video saved to {output_video1_path}")
        print(f"Camera 2 video saved to {output_video2_path}")
        print(f"3D skeleton video saved to {output_3d_video}")


def apply_anatomical_constraints(points_3d):
    """Apply anatomical constraints to ensure valid human skeletal structure"""
    # Handle empty or None input
    if points_3d is None or len(points_3d) == 0:
        return points_3d

    # Define expected bone lengths based on average proportions
    expected_bone_lengths = {
        (11, 13): 0.3,  # Upper arm
        (13, 15): 0.25,  # Lower arm
        (12, 14): 0.3,  # Upper arm
        (14, 16): 0.25,  # Lower arm
        (11, 12): 0.35,  # Shoulder width
        (23, 24): 0.25,  # Hip width
        (11, 23): 0.55,  # Left shoulder to hip
        (12, 24): 0.55,  # Right shoulder to hip
        (23, 25): 0.45,  # Left hip to knee
        (24, 26): 0.45,  # Right hip to knee
        (25, 27): 0.45,  # Left knee to ankle
        (26, 28): 0.45   # Right knee to ankle
    }

    # Make a copy to avoid modifying the original during iterations
    constrained_points = points_3d.copy()

    # Enforce bone length constraints
    for (idx1, idx2), expected_length in expected_bone_lengths.items():
        if idx1 < len(constrained_points) and idx2 < len(constrained_points):
            actual_vec = constrained_points[idx2] - constrained_points[idx1]
            actual_length = np.linalg.norm(actual_vec)

            # Skip if actual_length is zero or very small to avoid division by zero
            if actual_length < 1e-6:
                continue

            # Only adjust if the length is significantly different
            if abs(actual_length - expected_length) > 0.1 * expected_length:
                # Scale vector to expected length
                scaled_vec = actual_vec * (expected_length / actual_length)

                # Adjust the points (distribute the correction 50/50)
                midpoint = (constrained_points[idx1] + constrained_points[idx2]) / 2
                constrained_points[idx1] = midpoint - scaled_vec / 2
                constrained_points[idx2] = midpoint + scaled_vec / 2

    # Enforce joint angle constraints
    # Define expected angle ranges for major joints
    angle_constraints = {
        # (joint1, pivot, joint2): (min_angle, max_angle)
        (11, 13, 15): (45, 175),  # Left elbow
        (12, 14, 16): (45, 175),  # Right elbow
        (23, 25, 27): (100, 180), # Left knee
        (24, 26, 28): (100, 180), # Right knee
    }

    # Apply angle constraints
    for (idx1, pivot_idx, idx2), (min_angle, max_angle) in angle_constraints.items():
        if (idx1 < len(constrained_points) and 
            pivot_idx < len(constrained_points) and 
            idx2 < len(constrained_points)):

            v1 = constrained_points[idx1] - constrained_points[pivot_idx]
            v2 = constrained_points[idx2] - constrained_points[pivot_idx]

            # Calculate the angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure it's in valid range
            angle = np.degrees(np.arccos(cos_angle))

            # If angle is outside the constraint range, adjust it
            if angle < min_angle:
                # Adjust to minimum angle
                target_angle = min_angle
                adjust = True
            elif angle > max_angle:
                # Adjust to maximum angle
                target_angle = max_angle
                adjust = True
            else:
                adjust = False

            if adjust:
                # Convert target angle to radians
                target_rad = np.radians(target_angle)

                # Create rotation matrix to rotate v2 to target angle
                # This is a simplified approach and works best for 2D rotations
                # For full 3D, you would need a more complex rotation operation

                # Normalize vectors
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)

                # Get rotation axis (perpendicular to plane of v1, v2)
                rotation_axis = np.cross(v1_norm, v2_norm)
                if np.linalg.norm(rotation_axis) < 1e-6:
                    # If vectors are collinear, create a perpendicular vector
                    rotation_axis = np.array([1, 0, 0]) if abs(v1_norm[1]) > abs(v1_norm[0]) else np.array([0, 1, 0])
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

                # Calculate adjustment angle
                adjustment_angle = target_rad - np.arccos(cos_angle)

                # Create rotation matrix
                from scipy.spatial.transform import Rotation as R
                rot = R.from_rotvec(rotation_axis * adjustment_angle)

                # Apply rotation to v2
                rotated_v2 = rot.apply(v2)

                # Update point position
                constrained_points[idx2] = constrained_points[pivot_idx] + rotated_v2

    return constrained_points


def smooth_trajectories(current_points, previous_points, alpha=0.7):
    """Apply temporal smoothing to reduce jitter"""
    # Handle edge cases
    if current_points is None or previous_points is None:
        return current_points

    if len(current_points) != len(previous_points):
        # If sizes don't match, pad the smaller one (usually just return current)
        return current_points

    # Simple exponential smoothing
    try:
        # Convert to appropriate type to ensure mathematical operations work
        current_np = np.array(current_points, dtype=np.float32)
        previous_np = np.array(previous_points, dtype=np.float32)

        smoothed_points = alpha * current_np + (1 - alpha) * previous_np
        return smoothed_points
    except Exception as e:
        print(f"Error during trajectory smoothing: {e}")
        # Return original points if smoothing fails
        return current_points


def visualize_3d_skeleton(points_3d, joint_connections, frame_num):
    """Visualize the 3D skeleton with proper joint connections and standing on ground"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Skip visualization if we don't have enough points
    if points_3d is None or len(points_3d) < 33:  # MediaPipe has 33 landmarks
        print(f"Frame {frame_num}: Not enough points to visualize ({len(points_3d) if points_3d is not None else 0} points)")
        return

    # Create figure for 3D visualization
    plt.close('all')  # Close any existing figures
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize points for better visualization
    points_np = np.array(points_3d)

    # Find the lowest points (feet) to establish the ground plane
    foot_indices = [27, 28, 29, 30, 31, 32]  # MediaPipe foot indices (ankles, heels, toes)
    valid_foot_indices = [idx for idx in foot_indices if idx < len(points_np)]

    if valid_foot_indices:
        feet_points = points_np[valid_foot_indices]
        lowest_y = np.min(feet_points[:, 1])
    else:
        # If no feet visible, use the overall lowest point
        lowest_y = np.min(points_np[:, 1])

    # X and Z center for horizontal alignment
    x_mean = np.mean(points_np[:, 0])
    z_mean = np.mean(points_np[:, 2])

    # Align the skeleton: center X and Z, place feet on the ground (Y=0)
    centered_points = points_np.copy()
    centered_points[:, 0] = points_np[:, 0] - x_mean  # Center X
    centered_points[:, 1] = points_np[:, 1] - lowest_y  # Bottom at Y=0 (ground)
    centered_points[:, 2] = points_np[:, 2] - z_mean  # Center Z

    # Get scale factor (height of the skeleton)
    if 0 in centered_points:  # Nose landmark
        head_y = centered_points[0, 1]
        # Scale the skeleton to have a reasonable height
        target_height = 1.6  # Approx human height
        scale = target_height / head_y if head_y > 0 else 1.0
    else:
        # If no head visible, use max distance as scale reference
        max_dist = np.max(np.linalg.norm(centered_points, axis=1))
        scale = 1.0 / max_dist if max_dist > 0 else 1.0

    # Scale the points
    scaled_points = centered_points * scale

    # Draw a floor grid for reference (X-Z plane at Y=0)
    grid_size = 2.0
    x_grid = np.linspace(-grid_size, grid_size, 10)
    z_grid = np.linspace(-grid_size, grid_size, 10)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)

    ax.plot_surface(X, Y, Z, alpha=0.2, color='gray')

    # Draw the skeleton connections
    # Use the joint_connections parameter to draw lines between joints
    for start_idx, end_idx in joint_connections:
        if start_idx < len(scaled_points) and end_idx < len(scaled_points):
            # Skip if either point is invalid (all zeros or NaN)
            if (np.allclose(scaled_points[start_idx], 0) or 
                np.allclose(scaled_points[end_idx], 0) or
                np.isnan(scaled_points[start_idx]).any() or 
                np.isnan(scaled_points[end_idx]).any()):
                continue

            xs = [scaled_points[start_idx, 0], scaled_points[end_idx, 0]]
            ys = [scaled_points[start_idx, 1], scaled_points[end_idx, 1]]
            zs = [scaled_points[start_idx, 2], scaled_points[end_idx, 2]]

            # Determine color based on body part
            color = 'red'  # Default color

            # Torso connections
            if (start_idx, end_idx) in [(11, 12), (12, 24), (24, 23), (23, 11)]:
                color = 'red'
            # Left arm connections
            elif start_idx == 11 or end_idx == 11 or start_idx in [13, 15, 17, 19] or end_idx in [13, 15, 17, 19]:
                color = 'blue'
            # Right arm connections
            elif start_idx == 12 or end_idx == 12 or start_idx in [14, 16, 18, 20] or end_idx in [14, 16, 18, 20]:
                color = 'green'
            # Left leg connections
            elif start_idx == 23 or end_idx == 23 or start_idx in [25, 27, 29, 31] or end_idx in [25, 27, 29, 31]:
                color = 'purple'
            # Right leg connections
            elif start_idx == 24 or end_idx == 24 or start_idx in [26, 28, 30, 32] or end_idx in [26, 28, 30, 32]:
                color = 'orange'

            ax.plot(xs, ys, zs, c=color, linewidth=3)

    # Plot keypoints with smaller markers for better visualization
    ax.scatter(scaled_points[:, 0], scaled_points[:, 1], scaled_points[:, 2], 
               color='black', s=20, marker='o')

    # Plot important joints with larger markers
    important_joints = [0, 11, 12, 23, 24]  # nose, shoulders, hips
    if all(idx < len(scaled_points) for idx in important_joints):
        ax.scatter(scaled_points[important_joints, 0], 
                   scaled_points[important_joints, 1], 
                   scaled_points[important_joints, 2], 
                   color='yellow', s=50, marker='o')

    # Add annotations for better understanding
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Pose Estimation - Frame {frame_num}')

    # Set fixed axis limits for consistency
    ax.set_xlim([-grid_size, grid_size])
    ax.set_ylim([0, grid_size*1.5])  # Make Y start from 0 (ground)
    ax.set_zlim([-grid_size, grid_size])

    # Set the camera view angle for better visualization
    ax.view_init(elev=15, azim=45)

    # Add a text annotation with frame number
    ax.text2D(0.05, 0.95, f"Frame: {frame_num}", transform=ax.transAxes)

    # Save the visualization as an image
    output_dir = '3d_reconstruction'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/frame_{frame_num:04d}.png', dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
