from copy import deepcopy

import numpy as np


def _observed_joint_names(frames3d):
    names = set()
    for frame in frames3d:
        names.update(frame.get("joints", {}).keys())
    return sorted(names)


def _joint_xyz(frame, joint_name):
    joint = frame.get("joints", {}).get(joint_name)
    if joint is None:
        return None
    xyz = np.array([joint["x"], joint["y"], joint["z"]], dtype=np.float64)
    if not np.all(np.isfinite(xyz)):
        return None
    return xyz


def _contiguous_observed_runs(values):
    runs = []
    start = None
    for index, value in enumerate(values):
        if value is not None and start is None:
            start = index
        elif value is None and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _savgol_sequence(sequence, window_length, polyorder):
    smoothed = np.array(sequence, dtype=np.float64, copy=True)
    half_window = window_length // 2

    for index in range(len(sequence)):
        start = max(0, index - half_window)
        end = min(len(sequence), index + half_window + 1)
        segment = sequence[start:end]
        offsets = np.arange(start - index, end - index, dtype=np.float64)
        degree = min(polyorder, len(segment) - 1)

        if degree <= 0:
            continue

        for axis in range(3):
            coefficients = np.polyfit(offsets, segment[:, axis], degree)
            smoothed[index, axis] = float(np.polyval(coefficients, 0.0))

    return smoothed


def smooth_3d_frames(frames3d, window_length=7, polyorder=2):
    """
    Smooth observed 3D joint trajectories with a Savitzky-Golay polynomial fit.

    Missing joints remain missing. Only contiguous observed runs long enough for
    the smoothing window are modified.
    """
    output_frames = deepcopy(frames3d)
    frame_count = len(output_frames)
    if frame_count < 3:
        return {
            "available": False,
            "method": None,
            "frames": output_frames,
            "window_length": None,
            "polyorder": None,
            "smoothed_joint_values": 0,
            "notes": "Not enough frames for smoothing.",
        }

    window_length = int(window_length)
    polyorder = int(polyorder)
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, frame_count if frame_count % 2 == 1 else frame_count - 1)

    if window_length <= polyorder or window_length < 3:
        return {
            "available": False,
            "method": None,
            "frames": output_frames,
            "window_length": None,
            "polyorder": None,
            "smoothed_joint_values": 0,
            "notes": "Not enough frames for smoothing.",
        }

    smoothed_values = 0
    for joint_name in _observed_joint_names(output_frames):
        values = [_joint_xyz(frame, joint_name) for frame in output_frames]
        for start, end in _contiguous_observed_runs(values):
            if end - start < window_length:
                continue

            sequence = np.array(values[start:end], dtype=np.float64)
            smoothed_sequence = _savgol_sequence(sequence, window_length, polyorder)
            for local_index, xyz in enumerate(smoothed_sequence):
                frame_index = start + local_index
                joint = output_frames[frame_index]["joints"][joint_name]
                if joint is None:
                    continue
                joint["x"] = float(xyz[0])
                joint["y"] = float(xyz[1])
                joint["z"] = float(xyz[2])
                joint["observed"] = True
                joint["smoothed"] = True
                smoothed_values += 1

    if smoothed_values == 0:
        return {
            "available": False,
            "method": None,
            "frames": output_frames,
            "window_length": window_length,
            "polyorder": polyorder,
            "smoothed_joint_values": 0,
            "notes": "No contiguous observed joint trajectories were long enough for smoothing.",
        }

    return {
        "available": True,
        "method": "savitzky_golay",
        "frames": output_frames,
        "window_length": window_length,
        "polyorder": polyorder,
        "smoothed_joint_values": smoothed_values,
        "notes": "Savitzky-Golay smoothing applied only to observed 3D joint trajectories.",
    }
