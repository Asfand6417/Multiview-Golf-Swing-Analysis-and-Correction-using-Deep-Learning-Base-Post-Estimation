import unittest

from backend.cv.smoothing import smooth_3d_frames


def _frame(index, x_value):
    return {
        "frame_index": index,
        "timestamp": index / 30.0,
        "joints": {
            "nose": {"x": x_value, "y": 2.0 * x_value, "z": 3.0, "observed": True},
            "right_wrist": None,
        },
    }


class SmoothingTests(unittest.TestCase):
    def test_smoothing_unavailable_when_not_enough_frames(self):
        result = smooth_3d_frames([_frame(0, 1.0), _frame(1, 2.0)])

        self.assertFalse(result["available"])
        self.assertEqual(result["notes"], "Not enough frames for smoothing.")

    def test_smoothing_skips_missing_joints(self):
        result = smooth_3d_frames([_frame(index, float(index)) for index in range(7)])

        self.assertTrue(result["available"])
        self.assertIsNone(result["frames"][0]["joints"]["right_wrist"])
        self.assertTrue(result["frames"][0]["joints"]["nose"]["smoothed"])


if __name__ == "__main__":
    unittest.main()
