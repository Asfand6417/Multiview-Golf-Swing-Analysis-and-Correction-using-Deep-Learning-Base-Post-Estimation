import unittest

from backend.cv.phase_segmentation import EIGHT_PHASES, segment_phases_with_svm


class PhaseSegmentationTests(unittest.TestCase):
    def test_phase_segmentation_unavailable_without_model(self):
        result = segment_phases_with_svm([], model_path="/tmp/no-such-model/swing_phase_svm.joblib")

        self.assertFalse(result["available"])
        self.assertIsNone(result["method"])
        self.assertEqual(result["phases"], [])
        self.assertEqual(result["notes"], "No trained SVM model found.")

    def test_phase_vocabulary_includes_transition(self):
        self.assertEqual(EIGHT_PHASES[4], "Transition")
        self.assertEqual(EIGHT_PHASES[-1], "Follow-through")


if __name__ == "__main__":
    unittest.main()
