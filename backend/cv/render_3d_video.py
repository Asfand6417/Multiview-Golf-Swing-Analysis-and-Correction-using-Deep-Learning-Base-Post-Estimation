import os
from pathlib import Path

matplotlib_config_dir = Path("/tmp/golfpose-matplotlib")
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BONES = [
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _observed_points(frames3d):
    points = []
    for frame in frames3d:
        for joint in frame.get("joints", {}).values():
            if joint is not None:
                points.append([joint["x"], joint["y"], joint["z"]])
    return np.array(points, dtype=np.float64) if points else np.empty((0, 3))


def render_3d_skeleton_video(frames3d, output_path, fps=24):
    if not frames3d:
        raise ValueError("No triangulated frames available for rendering.")

    observed = _observed_points(frames3d)
    if observed.size == 0:
        raise ValueError("No observed 3D joints available for rendering.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mins = observed.min(axis=0)
    maxs = observed.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)

    width, height = 800, 600
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError("Could not open MP4 writer for rendered 3D video.")

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    try:
        for frame in frames3d:
            ax.clear()
            ax.set_title(f"Frame {frame.get('frame_index')}")
            ax.set_xlim(centers[0] - radius, centers[0] + radius)
            ax.set_ylim(centers[1] - radius, centers[1] + radius)
            ax.set_zlim(centers[2] - radius, centers[2] + radius)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            joints = frame.get("joints", {})
            frame_points = []
            for joint in joints.values():
                if joint is not None:
                    frame_points.append([joint["x"], joint["y"], joint["z"]])
            if frame_points:
                pts = np.array(frame_points, dtype=np.float64)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#75ff9e", s=28)

            for start, end in BONES:
                start_joint = joints.get(start)
                end_joint = joints.get(end)
                if start_joint is None or end_joint is None:
                    continue
                ax.plot(
                    [start_joint["x"], end_joint["x"]],
                    [start_joint["y"], end_joint["y"]],
                    [start_joint["z"], end_joint["z"]],
                    c="#75ff9e",
                    linewidth=2,
                )

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()
        plt.close(fig)

    return str(output_path)
