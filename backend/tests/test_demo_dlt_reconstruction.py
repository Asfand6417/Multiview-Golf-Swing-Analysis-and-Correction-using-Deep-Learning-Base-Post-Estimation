import asyncio
import json
import unittest
from pathlib import Path

from backend.main import DEMO_FIXTURE_PATH, get_demo_dlt_reconstruction


class DemoDltReconstructionTests(unittest.TestCase):
    def setUp(self):
        with Path(DEMO_FIXTURE_PATH).open("r", encoding="utf-8") as fixture_file:
            self.fixture = json.load(fixture_file)

    def test_fixture_matches_reconstruction_contract(self):
        joint_names = self.fixture["joint_names"]
        self.assertGreater(len(joint_names), 0)

        for camera_name, matrix in self.fixture["projection_matrices"].items():
            with self.subTest(camera=camera_name):
                self.assertEqual(len(matrix), 3)
                self.assertTrue(all(len(row) == 4 for row in matrix))

        timestamps = []
        for frame in self.fixture["frames"]:
            timestamps.append(frame["timestamp"])
            self.assertEqual(len(frame["points_front"]), len(joint_names))
            self.assertEqual(len(frame["points_side"]), len(joint_names))

        self.assertEqual(timestamps, sorted(timestamps))

    def test_demo_endpoint_returns_raw_dlt_frames(self):
        response = asyncio.run(get_demo_dlt_reconstruction())

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["source"], "calibrated_demo")
        self.assertGreater(response["frame_count"], 0)
        self.assertEqual(response["frame_count"], len(response["frames"]))
        self.assertGreater(len(response["joint_names"]), 0)
        self.assertFalse(response["metrics"]["mpjpe_available"])
        self.assertIsNone(response["metrics"]["mpjpe_mm"])

        missing_joint_frame = response["frames"][3]["joints"]
        self.assertIsNone(missing_joint_frame["right_wrist"])
        self.assertIsNotNone(response["frames"][3]["joints"]["left_wrist"])


if __name__ == "__main__":
    unittest.main()
