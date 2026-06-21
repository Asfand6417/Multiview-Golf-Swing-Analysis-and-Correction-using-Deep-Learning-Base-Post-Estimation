import unittest

from backend.cv.biomechanics import analyze_biomechanics


class BiomechanicsTests(unittest.TestCase):
    def test_biomechanics_unavailable_when_features_are_missing(self):
        result = analyze_biomechanics([
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "features": {
                    "x_factor_deg": None,
                    "spine_angle_deg": None,
                },
            }
        ], {"available": True, "phases": [{"frame_index": 0, "phase": "Top"}]})

        self.assertFalse(result["available"])
        self.assertEqual(result["errors"], [])

    def test_biomechanics_requires_phase_labels(self):
        result = analyze_biomechanics([
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "features": {
                    "x_factor_deg": 20.0,
                    "spine_angle_deg": 30.0,
                },
            }
        ], {"available": False, "phases": []})

        self.assertFalse(result["available"])
        self.assertIn("phase labels", result["notes"])


if __name__ == "__main__":
    unittest.main()
