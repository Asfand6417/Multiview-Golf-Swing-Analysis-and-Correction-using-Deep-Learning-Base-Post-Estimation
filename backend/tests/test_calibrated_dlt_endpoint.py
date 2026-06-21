import asyncio
from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend.main import run_calibrated_dlt_analysis


def _pose_payload():
    return {
        "fps": 30.0,
        "frame_count": 1,
        "joint_names": ["nose", "left_wrist", "right_wrist"],
        "frames": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "points": {
                    "nose": [640.0, 755.122],
                    "left_wrist": [493.535, 572.121],
                    "right_wrist": None,
                },
                "visibility": {},
            }
        ],
    }


class CalibratedDltEndpointTests(unittest.TestCase):
    def test_analysis_rejects_missing_calibration_id(self):
        with self.assertRaises(HTTPException) as context:
            asyncio.run(run_calibrated_dlt_analysis(
                front_video=_upload("front.mp4"),
                side_video=_upload("side.mp4"),
                calibration_id="",
            ))

        self.assertEqual(context.exception.status_code, 422)

    def test_analysis_never_returns_mpjpe_without_ground_truth(self):
        P_front = np.array([[1000.0, 0.0, 640.0, 0.0], [0.0, 1000.0, 360.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        P_side = np.array([[1000.0, 0.0, 640.0, 500.0], [0.0, 1000.0, 360.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        calibration = {"calibration_id": "cal-1", "P_front": P_front.tolist(), "P_side": P_side.tolist()}

        with (
            patch("backend.main._load_calibration", return_value=(calibration, P_front, P_side)),
            patch("backend.main.extract_pose_keypoints", return_value=_pose_payload()),
            patch("backend.main.render_3d_skeleton_video", return_value="/tmp/rendered.mp4"),
        ):
            payload = asyncio.run(run_calibrated_dlt_analysis(
                front_video=_upload("front.mp4"),
                side_video=_upload("side.mp4"),
                calibration_id="cal-1",
            ))

        self.assertEqual(payload["source"], "uploaded_calibrated_dlt")
        self.assertFalse(payload["metrics"]["mpjpe_available"])
        self.assertIsNone(payload["metrics"]["mpjpe_mm"])
        self.assertIsNone(payload["frames"][0]["joints"]["right_wrist"])
        self.assertEqual(payload["raw_frames"], payload["smoothed_frames"])
        self.assertFalse(payload["smoothing"]["available"])
        self.assertEqual(payload["smoothing"]["notes"], "Not enough frames for smoothing.")
        self.assertFalse(payload["phase_segmentation"]["available"])


def _upload(filename):
    return UploadFile(BytesIO(b"fake"), filename=filename)


if __name__ == "__main__":
    unittest.main()
