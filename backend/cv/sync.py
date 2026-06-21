def sync_two_view_keypoints(front_frames, side_frames):
    """Pair front and side frames by nearest timestamp without inventing quality scores."""
    if not front_frames or not side_frames:
        return {
            "paired_frames": [],
            "sync_quality": None,
            "notes": "No frames available from one or both views.",
        }

    paired_frames = []
    side_sorted = sorted(side_frames, key=lambda frame: frame.get("timestamp", 0.0))

    for front_frame in sorted(front_frames, key=lambda frame: frame.get("timestamp", 0.0)):
        front_timestamp = float(front_frame.get("timestamp", 0.0))
        side_frame = min(
            side_sorted,
            key=lambda frame: abs(float(frame.get("timestamp", 0.0)) - front_timestamp),
        )
        side_timestamp = float(side_frame.get("timestamp", 0.0))
        paired_frames.append({
            "frame_index": front_frame.get("frame_index"),
            "timestamp": front_timestamp,
            "front": front_frame,
            "side": side_frame,
            "timestamp_delta": abs(front_timestamp - side_timestamp),
        })

    return {
        "paired_frames": paired_frames,
        "sync_quality": None,
        "notes": "Frames matched by nearest timestamp; no independent sync quality metric computed.",
    }
