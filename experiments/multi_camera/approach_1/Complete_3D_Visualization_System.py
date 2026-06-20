
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Remove this line - it's causing the circular import
# from multicamera.Complete_3D_Visualization_System import create_3d_video
# Instead, import only what works
# from multicamera.Complete_3D_Visualization_System import create_3d_video
from multicamera.Joint_Position_Optimization_and_Confidence_Based_Integration import optimize_keypoints
from multicamera.camera_calibration import calibrate_cameras
from multicamera.camera_setup import setup_cameras
from multicamera.multiview_poseestimation import extract_keypoints


def create_3d_video(output_file, fps=30):
    """Create a video from the saved 3D visualization frames with improved debugging"""
    import cv2
    import os
    import glob
    
    frame_dir = '3d_reconstruction'
    frame_files = sorted(glob.glob(f'{frame_dir}/frame_*.png'))
    
    if not frame_files:
        print("No frames found for video creation")
        return
    
    print(f"Found {len(frame_files)} frame files for video creation")
    print(f"First frame: {frame_files[0]}")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Error: Could not read first frame file: {frame_files[0]}")
        return
        
    h, w, c = first_frame.shape
    print(f"Frame dimensions: {w}x{h}, {c} channels")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print(f"Error: Could not create video writer for file: {output_file}")
        return
    
    # Add each frame to the video
    frames_added = 0
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            out.write(frame)
            frames_added += 1
        else:
            print(f"Warning: Could not read frame file: {frame_file}")
    
    # Release resources
    out.release()
    print(f"3D visualization video created with {frames_added} frames: {output_file}")

def visualize_3d_skeleton(points_3d, joint_connections, frame_num):
    """Visualize the 3D skeleton with proper joint connections with direct display"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Skip visualization if we don't have enough points
    if points_3d is None or len(points_3d) == 0:
        print(f"Frame {frame_num}: No points to visualize")
        return
    
    # Print debug info about the points
    print(f"Frame {frame_num}: Visualizing {len(points_3d)} 3D points")
    print(f"Sample points: {points_3d[0:3]}")
    
    # Create figure for 3D visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all keypoints
    xs = [p[0] for p in points_3d]
    ys = [p[1] for p in points_3d]
    zs = [p[2] for p in points_3d]
    ax.scatter(xs, ys, zs, c='blue', marker='o', s=20)
    
    # Connect joints with lines according to the skeleton structure
    for start_idx, end_idx in joint_connections:
        if start_idx < len(points_3d) and end_idx < len(points_3d):
            start_point = points_3d[start_idx]
            end_point = points_3d[end_idx]
            
            # Skip if either point is invalid (all zeros from a placeholder)
            if (np.allclose(start_point, 0) or np.allclose(end_point, 0) or
                np.isnan(start_point).any() or np.isnan(end_point).any()):
                continue
                
            ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    [start_point[2], end_point[2]], 'r-', linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Skeleton Reconstruction - Frame {frame_num}')
    
    # Set view limits
    ax_range = 1.0  # Adjust based on your data scale
    ax.set_xlim([-ax_range, ax_range])
    ax.set_ylim([-ax_range, ax_range])
    ax.set_zlim([-ax_range, ax_range])
    
    # Save figure to file
    output_dir = '3d_reconstruction'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/frame_{frame_num:04d}.png')
    
    # Show plot - comment this for production
    plt.show(block=False)
    plt.pause(0.01)  # Short pause to display
    plt.close(fig)

# Add the missing function here
# def visualize_3d_skeleton(points_3d, joint_connections, frame_num):
#     """
#     Visualize 3D skeleton and save the visualization as an image.
    
#     Args:
#         points_3d: Array of 3D joint positions
#         joint_connections: List of tuples defining connections between joints
#         frame_num: Frame number for saving the visualization
#     """
#     # Create a new figure
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Extract x, y, z coordinates
#     xs = points_3d[:, 0]
#     ys = points_3d[:, 1]
#     zs = points_3d[:, 2]
    
#     # Plot all joints as scatter points
#     ax.scatter(xs, ys, zs, c='blue', marker='o', s=50)
    
#     # Draw lines between connected joints
#     for start_idx, end_idx in joint_connections:
#         if start_idx < len(points_3d) and end_idx < len(points_3d):
#             x_pair = [points_3d[start_idx, 0], points_3d[end_idx, 0]]
#             y_pair = [points_3d[start_idx, 1], points_3d[end_idx, 1]]
#             z_pair = [points_3d[start_idx, 2], points_3d[end_idx, 2]]
#             ax.plot(x_pair, y_pair, z_pair, 'r-', linewidth=2)
    
#     # Set axis limits (adjust these based on your coordinate system)
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])
    
#     # Set labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(f'3D Pose Reconstruction - Frame {frame_num}')
    
#     # Save figure
#     output_dir = '3d_reconstruction'
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, f'frame_{frame_num:04d}.png'))
    
#     # Close the plot to avoid memory issues
#     plt.close(fig)


def triangulate_points(keypoints1_opt, keypoints2_opt, calibration_data):
    """Triangulate 3D points with enhanced debugging"""
    
    print(f"Input shape - keypoints1: {keypoints1_opt.shape}, keypoints2: {keypoints2_opt.shape}")
    
    # Get calibration parameters
    mtx1 = calibration_data['camera1_matrix']
    mtx2 = calibration_data['camera2_matrix']
    R = calibration_data['R']
    T = calibration_data['T']
    
    # Print calibration data to check validity
    print(f"Camera 1 matrix shape: {mtx1.shape}")
    print(f"Camera 2 matrix shape: {mtx2.shape}")
    print(f"Rotation matrix shape: {R.shape}")
    print(f"Translation vector shape: {T.shape}")
    
    # Compute projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = mtx1 @ P1
    
    RT = np.hstack((R, T))
    P2 = mtx2 @ RT
    
    print(f"P1 shape: {P1.shape}, P2 shape: {P2.shape}")
    
    # Log a few keypoints for debugging
    for i in range(min(3, len(keypoints1_opt))):
        print(f"Keypoint pair {i}: Camera 1: {keypoints1_opt[i]}, Camera 2: {keypoints2_opt[i]}")
    
    # Initialize array for 3D points
    points_3d = []
    
    # Count valid triangulations
    valid_count = 0
    
    for i, (kp1, kp2) in enumerate(zip(keypoints1_opt, keypoints2_opt)):
        try:
            # Convert points to correct format
            point1 = np.array([[kp1[0]], [kp1[1]]], dtype=np.float32)
            point2 = np.array([[kp2[0]], [kp2[1]]], dtype=np.float32)
            
            # Triangulate one point
            point_4d = cv2.triangulatePoints(P1, P2, point1, point2)
            
            # Convert to 3D homogeneous coordinates
            point_3d = point_4d[:3] / point_4d[3]
            
            # Check for valid triangulation
            if not np.isnan(point_3d).any() and not np.isinf(point_3d).any():
                points_3d.append(point_3d.flatten())
                valid_count += 1
            else:
                print(f"Invalid triangulation for point {i}: NaN or Inf values")
                points_3d.append(np.zeros(3))  # Placeholder
        except Exception as e:
            print(f"Error triangulating point {i}: {e}")
            points_3d.append(np.zeros(3))  # Placeholder
    
    result = np.array(points_3d)
    print(f"Triangulation complete: {valid_count}/{len(keypoints1_opt)} valid points")
    
    # Print stats about the triangulated points
    if len(result) > 0:
        print(f"3D points min: {np.min(result, axis=0)}")
        print(f"3D points max: {np.max(result, axis=0)}")
        print(f"3D points mean: {np.mean(result, axis=0)}")
    
    return result


def main():
    """Main execution function for dual-camera 3D reconstruction"""
    # Setup directory for 3D reconstruction frames
    os.makedirs('3d_reconstruction', exist_ok=True)

    # Step 1: Calibrate cameras
    if not os.path.exists('calibration_data.npy'):
        print("Performing camera calibration...")
        calibration_data = calibrate_cameras()
    else:
        print("Loading existing camera calibration...")
        calibration_data = np.load('calibration_data.npy', allow_pickle=True).item()

    # Step 2: Setup synchronized cameras
    cam1, cam2 = setup_cameras()

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
    prev_points_3d = None  # Initialize this variable for the first frame

    try:
        while True:
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()

            if not ret1 or not ret2:
                break

            # Extract keypoints from both views
            keypoints1, conf1, frame1_annotated = extract_keypoints(frame1)
            keypoints2, conf2, frame2_annotated = extract_keypoints(frame2)

            if len(keypoints1) == 0 or len(keypoints2) == 0:
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
            if frame_num > 0:
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
        cv2.destroyAllWindows()

        # Create 3D visualization video
        create_3d_video('3d_reconstruction.mp4')

        print(f"Processed {frame_num} frames successfully")


def apply_anatomical_constraints(points_3d):
    """Apply anatomical constraints to ensure valid human skeletal structure with improved error handling"""
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
            # Get vectors between joints
            p1 = constrained_points[idx1]
            p2 = constrained_points[idx2]
            
            # Skip if either point is invalid (all zeros from a placeholder)
            if np.allclose(p1, 0) or np.allclose(p2, 0):
                continue
                
            actual_vec = p2 - p1
            actual_length = np.linalg.norm(actual_vec)
            
            # Skip if actual_length is zero or very small to avoid division by zero
            if actual_length < 1e-6:
                continue

            # Only adjust if the length is significantly different
            if abs(actual_length - expected_length) > 0.1 * expected_length:
                # Scale vector to expected length
                scaled_vec = actual_vec * (expected_length / actual_length)

                # Adjust the points (distribute the correction 50/50)
                midpoint = (p1 + p2) / 2
                constrained_points[idx1] = midpoint - scaled_vec / 2
                constrained_points[idx2] = midpoint + scaled_vec / 2

    # Enforce joint angle constraints - with better error handling
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
            
            # Get joint positions
            p1 = constrained_points[idx1]
            pivot = constrained_points[pivot_idx]
            p2 = constrained_points[idx2]
            
            # Skip if any point is invalid (all zeros from a placeholder)
            if np.allclose(p1, 0) or np.allclose(pivot, 0) or np.allclose(p2, 0):
                continue
                
            # Calculate vectors from pivot to the other two points
            v1 = p1 - pivot
            v2 = p2 - pivot
            
            # Skip if either vector is too small
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                continue
            
            # Calculate the angle between vectors (safely)
            dot_product = np.dot(v1, v2)
            cos_angle = dot_product / (v1_norm * v2_norm)
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
                try:
                    # Convert target angle to radians
                    target_rad = np.radians(target_angle)
                    
                    # Get rotation axis (perpendicular to plane of v1, v2)
                    rotation_axis = np.cross(v1, v2)
                    rotation_axis_norm = np.linalg.norm(rotation_axis)
                    
                    if rotation_axis_norm < 1e-6:
                        # If vectors are collinear, create a perpendicular vector
                        rotation_axis = np.array([1, 0, 0]) if abs(v1[1]) > abs(v1[0]) else np.array([0, 1, 0])
                    else:
                        rotation_axis = rotation_axis / rotation_axis_norm
                    
                    # Calculate adjustment angle
                    adjustment_angle = target_rad - np.arccos(cos_angle)
                    
                    # Create rotation matrix
                    from scipy.spatial.transform import Rotation as R
                    rot = R.from_rotvec(rotation_axis * adjustment_angle)
                    
                    # Apply rotation to v2
                    rotated_v2 = rot.apply(v2)
                    
                    # Update point position
                    constrained_points[idx2] = pivot + rotated_v2
                except Exception as e:
                    print(f"Error in angle adjustment: {e}")
                    # Continue without adjustment if error occurs

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


def test_skeleton_visualization():
    """Standalone test that creates and displays a 3D skeleton visualization"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import os
    import time

    print("Starting standalone 3D skeleton test...")

    # MediaPipe 33 joints in a T-pose
    points = np.zeros((33, 3))  # Initialize all joints to origin

    # Add basic human proportions (simplified)
    # Head
    points[0] = [0, 0, 0]  # Nose

    # Left eye
    points[2] = [-0.1, 0.1, 0.1]  # Left eye inner
    points[3] = [-0.15, 0.12, 0.12]  # Left eye
    points[4] = [-0.2, 0.1, 0.1]  # Left eye outer

    # Right eye
    points[1] = [0.1, 0.1, 0.1]  # Right eye inner
    points[5] = [0.15, 0.12, 0.12]  # Right eye
    points[6] = [0.2, 0.1, 0.1]  # Right eye outer

    # Ears
    points[7] = [-0.3, 0, 0]  # Left ear
    points[8] = [0.3, 0, 0]  # Right ear

    # Mouth
    points[9] = [-0.05, -0.1, 0.1]  # Mouth left
    points[10] = [0.05, -0.1, 0.1]  # Mouth right

    # Shoulders
    points[11] = [-0.5, -0.3, 0]  # Left shoulder
    points[12] = [0.5, -0.3, 0]  # Right shoulder

    # Elbows
    points[13] = [-0.8, -0.3, 0]  # Left elbow
    points[14] = [0.8, -0.3, 0]  # Right elbow

    # Wrists
    points[15] = [-1.1, -0.3, 0]  # Left wrist
    points[16] = [1.1, -0.3, 0]  # Right wrist

    # Hands
    points[17] = [-1.2, -0.35, 0]  # Left pinky
    points[19] = [-1.2, -0.3, 0]  # Left index
    points[21] = [-1.3, -0.3, 0]  # Left thumb

    points[18] = [1.2, -0.35, 0]  # Right pinky
    points[20] = [1.2, -0.3, 0]  # Right index
    points[22] = [1.3, -0.3, 0]  # Right thumb

    # Torso
    points[23] = [-0.25, -1.0, 0]  # Left hip
    points[24] = [0.25, -1.0, 0]  # Right hip

    # Knees
    points[25] = [-0.3, -1.6, 0]  # Left knee
    points[26] = [0.3, -1.6, 0]  # Right knee

    # Ankles
    points[27] = [-0.35, -2.2, 0]  # Left ankle
    points[28] = [0.35, -2.2, 0]  # Right ankle

    # Feet
    points[29] = [-0.4, -2.3, 0.1]  # Left heel
    points[31] = [-0.4, -2.3, 0.2]  # Left foot index

    points[30] = [0.4, -2.3, 0.1]  # Right heel
    points[32] = [0.4, -2.3, 0.2]  # Right foot index

    # Define MediaPipe joint connections
    joint_connections = [
        # Face
        (0, 1), (1, 5), (5, 6), (0, 2), (2, 3), (3, 4),
        (0, 9), (9, 10), (10, 0),
        (4, 7), (7, 9), (6, 8), (8, 10),

        # Torso
        (11, 12), (12, 24), (24, 23), (23, 11),

        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),

        # Right arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),

        # Left leg
        (23, 25), (25, 27), (27, 29), (27, 31),

        # Right leg
        (24, 26), (26, 28), (28, 30), (28, 32)
    ]

    # Create output directory
    os.makedirs('test_skeleton', exist_ok=True)

    # Create frames showing skeleton from different angles
    for angle in range(0, 360, 10):
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot keypoints
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=20)

        # Draw skeleton connections with different colors by body part
        colors = {
            'face': 'cyan',
            'torso': 'red',
            'left_arm': 'blue',
            'right_arm': 'green',
            'left_leg': 'magenta',
            'right_leg': 'yellow'
        }

        # Map connections to body parts
        body_part_indices = {
            'face': list(range(11)),
            'torso': [11, 12, 23, 24],
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32]
        }

        # Draw connections
        for start_idx, end_idx in joint_connections:
            # Determine color based on which body part the joints belong to
            color = 'gray'  # Default color
            for part, indices in body_part_indices.items():
                if start_idx in indices and end_idx in indices:
                    color = colors[part]
                    break

            # Draw line
            ax.plot([points[start_idx, 0], points[end_idx, 0]],
                    [points[start_idx, 1], points[end_idx, 1]],
                    [points[start_idx, 2], points[end_idx, 2]],
                    color=color, linewidth=2)

        # Set consistent view limits
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-2.5, 0.5])
        ax.set_zlim([-1.0, 1.0])

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set title
        ax.set_title(f'3D Skeleton Test - Angle {angle}°')

        # Set viewing angle
        ax.view_init(elev=20, azim=angle)

        # Save figure
        plt.savefig(f'test_skeleton/skeleton_angle_{angle:03d}.png')

        # Show figure (for interactive testing)
        plt.pause(0.1)
        plt.close(fig)

    print("Skeleton test completed. Check test_skeleton folder for results.")

    # Create a video from the frames
    try:
        import cv2

        # Define video parameters
        fps = 15
        frame_path = 'test_skeleton/skeleton_angle_%03d.png'
        video_path = 'test_skeleton/skeleton_animation.mp4'

        # Read the first frame to get dimensions
        first_frame = cv2.imread('test_skeleton/skeleton_angle_000.png')
        if first_frame is not None:
            h, w, c = first_frame.shape

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

            # Add frames to video
            for angle in range(0, 360, 10):
                frame_file = f'test_skeleton/skeleton_angle_{angle:03d}.png'
                frame = cv2.imread(frame_file)
                if frame is not None:
                    out.write(frame)

            # Release resources
            out.release()
            print(f"Created test animation: {video_path}")
        else:
            print("Error: Could not read first frame")
    except Exception as e:
        print(f"Error creating video: {e}")
