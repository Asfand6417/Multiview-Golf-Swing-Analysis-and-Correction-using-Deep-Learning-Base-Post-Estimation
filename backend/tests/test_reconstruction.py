import unittest

import numpy as np

from backend.cv.phase_segmentation import segment_phases_with_svm
from backend.cv.reconstruction import triangulate_joint, triangulate_sequence
from backend.cv.render_3d_video import render_3d_skeleton_video


class ReconstructionTests(unittest.TestCase):
    def test_triangulate_joint_returns_null_for_missing_point(self):
        self.assertIsNone(triangulate_joint(None, [1.0, 2.0], np.zeros((3, 4)), np.ones((3, 4))))
        self.assertIsNone(triangulate_joint([1.0, 2.0], None, np.zeros((3, 4)), np.ones((3, 4))))

    def test_triangulate_sequence_preserves_missing_joints(self):
        P_front = np.array([[1000.0, 0.0, 640.0, 0.0], [0.0, 1000.0, 360.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        P_side = np.array([[1000.0, 0.0, 640.0, 500.0], [0.0, 1000.0, 360.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        paired_frames = [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "front": {"points": {"nose": [640.0, 755.122], "right_wrist": None}},
                "side": {"points": {"nose": [761.951, 755.122], "right_wrist": [893.589, 565.742]}},
            }
        ]

        frames = triangulate_sequence(paired_frames, ["nose", "right_wrist"], P_front, P_side)

        self.assertIsNotNone(frames[0]["joints"]["nose"])
        self.assertIsNone(frames[0]["joints"]["right_wrist"])

    def test_svm_segmentation_unavailable_without_real_model(self):
        result = segment_phases_with_svm([], model_path="/tmp/does-not-exist/swing_phase_svm.joblib")

        self.assertFalse(result["available"])
        self.assertIsNone(result["method"])
        self.assertEqual(result["phases"], [])

    def test_rendered_video_requires_observed_3d_joints(self):
        with self.assertRaises(ValueError):
            render_3d_skeleton_video([
                {
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "joints": {"nose": None},
                }
            ], "/tmp/unused_golfpose_render.mp4")


if __name__ == "__main__":
    unittest.main()
