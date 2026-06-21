import asyncio
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend.cv.calibration import build_projection_matrices
from backend.main import calibrate_stereo_endpoint


class CalibrationEndpointTests(unittest.TestCase):
    def test_calibration_rejects_mismatched_image_counts(self):
        with self.assertRaises(HTTPException) as context:
            asyncio.run(calibrate_stereo_endpoint(
                front_images=[_upload("front_0.png")],
                side_images=[_upload("side_0.png"), _upload("side_1.png")],
                board_rows=6,
                board_cols=9,
                square_size_mm=25,
            ))

        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("counts must match", context.exception.detail)

    def test_calibration_rejects_fewer_than_ten_valid_pairs(self):
        calibration = {
            "K_front": np.eye(3),
            "D_front": np.zeros((1, 5)),
            "K_side": np.eye(3),
            "D_side": np.zeros((1, 5)),
            "R": np.eye(3),
            "T": np.array([[1.0], [0.0], [0.0]]),
            "E": np.eye(3),
            "F": np.eye(3),
            "P_front": np.zeros((3, 4)),
            "P_side": np.ones((3, 4)),
            "reprojection_error": 0.5,
            "valid_pair_count": 9,
            "image_size": [640, 480],
        }
        with patch("backend.main.calibrate_stereo_cameras", return_value=calibration):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(calibrate_stereo_endpoint(
                    front_images=[_upload(f"front_{index}.png") for index in range(9)],
                    side_images=[_upload(f"side_{index}.png") for index in range(9)],
                    board_rows=6,
                    board_cols=9,
                    square_size_mm=25,
                ))

        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("At least 10 valid", context.exception.detail)

    def test_projection_matrices_are_three_by_four(self):
        P_front, P_side = build_projection_matrices(
            np.eye(3),
            np.eye(3),
            np.eye(3),
            np.array([[1.0], [0.0], [0.0]]),
        )

        self.assertEqual(P_front.shape, (3, 4))
        self.assertEqual(P_side.shape, (3, 4))

    def test_calibration_endpoint_returns_real_projection_matrices_from_checkerboards(self):
        front_uploads, side_uploads = _checkerboard_upload_pairs()

        with tempfile.TemporaryDirectory() as calibration_dir:
            with patch("backend.main.CALIBRATIONS_DIR", Path(calibration_dir)):
                payload = asyncio.run(calibrate_stereo_endpoint(
                    front_images=front_uploads,
                    side_images=side_uploads,
                    board_rows=6,
                    board_cols=9,
                    square_size_mm=25,
                ))

            saved_file = Path(calibration_dir) / f"{payload['calibration_id']}.json"
            saved_payload = json.loads(saved_file.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["valid_pair_count"], 12)
        self.assertEqual(np.asarray(payload["P_front"]).shape, (3, 4))
        self.assertEqual(np.asarray(payload["P_side"]).shape, (3, 4))
        self.assertGreater(np.linalg.norm(np.asarray(payload["T"], dtype=np.float64)), 0.0)
        self.assertEqual(saved_payload["P_front"], payload["P_front"])
        self.assertEqual(saved_payload["P_side"], payload["P_side"])


def _upload(filename):
    return UploadFile(BytesIO(b"fake"), filename=filename)


def _checkerboard_png_bytes(board_rows, board_cols, tx, ty, angle_degrees):
    square_size = 36
    image_height, image_width = 520, 760
    pattern_width = (board_cols + 1) * square_size
    pattern_height = (board_rows + 1) * square_size
    board = np.full((pattern_height, pattern_width), 255, np.uint8)

    for y_index in range(board_rows + 1):
        for x_index in range(board_cols + 1):
            if (x_index + y_index) % 2 == 0:
                cv2.rectangle(
                    board,
                    (x_index * square_size, y_index * square_size),
                    ((x_index + 1) * square_size, (y_index + 1) * square_size),
                    0,
                    -1,
                )

    canvas = np.full((image_height, image_width), 180, np.uint8)
    matrix = cv2.getRotationMatrix2D((pattern_width / 2, pattern_height / 2), angle_degrees, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    warped = cv2.warpAffine(board, matrix, (image_width, image_height), borderValue=180)
    canvas[warped != 180] = warped[warped != 180]
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise AssertionError("Could not encode generated checkerboard image.")
    return encoded.tobytes()


def _checkerboard_upload_pairs():
    front_uploads = []
    side_uploads = []
    for index in range(12):
        tx = 90 + (index % 4) * 30
        ty = 70 + (index // 4) * 35
        angle = (index - 5) * 1.8
        front_uploads.append(UploadFile(
            BytesIO(_checkerboard_png_bytes(6, 9, tx, ty, angle)),
            filename=f"front_{index}.png",
        ))
        side_uploads.append(UploadFile(
            BytesIO(_checkerboard_png_bytes(6, 9, tx + 25, ty + 4, angle + 0.5)),
            filename=f"side_{index}.png",
        ))
    return front_uploads, side_uploads


if __name__ == "__main__":
    unittest.main()
