import json
import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uuid

os.makedirs("/tmp/golfpose-matplotlib", exist_ok=True)
os.environ["MPLCONFIGDIR"] = "/tmp/golfpose-matplotlib"

try:
    from cv.biomechanics import analyze_biomechanics
    from cv.calibration import CalibrationError, calibration_to_json, calibrate_stereo_cameras, save_calibration
    from cv.features import extract_biomechanical_features
    from cv.phase_segmentation import extract_phase_features, segment_phases_with_svm
    from cv.pose_extraction import extract_pose_keypoints
    from cv.reconstruction import reconstruct_dlt_frames, triangulate_sequence
    from cv.render_3d_video import render_3d_skeleton_video
    from cv.smoothing import smooth_3d_frames
    from cv.sync import sync_two_view_keypoints
except ModuleNotFoundError:
    from backend.cv.biomechanics import analyze_biomechanics
    from backend.cv.calibration import CalibrationError, calibration_to_json, calibrate_stereo_cameras, save_calibration
    from backend.cv.features import extract_biomechanical_features
    from backend.cv.phase_segmentation import extract_phase_features, segment_phases_with_svm
    from backend.cv.pose_extraction import extract_pose_keypoints
    from backend.cv.reconstruction import reconstruct_dlt_frames, triangulate_sequence
    from backend.cv.render_3d_video import render_3d_skeleton_video
    from backend.cv.smoothing import smooth_3d_frames
    from backend.cv.sync import sync_two_view_keypoints

app = FastAPI(title="GolfPose API")
DEMO_FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "calibrated_dlt_demo.json"
CALIBRATIONS_DIR = Path(__file__).resolve().parent / "data" / "calibrations"
JOBS_DIR = Path(__file__).resolve().parent / "data" / "jobs"
RENDERED_DIR = Path(__file__).resolve().parent / "data" / "rendered"
for data_dir in (CALIBRATIONS_DIR, JOBS_DIR, RENDERED_DIR):
    data_dir.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _save_upload(upload: UploadFile, output_path: Path):
    with output_path.open("wb") as file_handle:
        file_handle.write(await upload.read())
    return output_path


def _load_calibration(calibration_id: str):
    if not calibration_id:
        raise HTTPException(status_code=422, detail="calibration_id is required.")

    calibration_path = CALIBRATIONS_DIR / f"{calibration_id}.json"
    if not calibration_path.exists():
        raise HTTPException(status_code=404, detail="Calibration ID was not found.")

    with calibration_path.open("r", encoding="utf-8") as file_handle:
        calibration = json.load(file_handle)

    P_front = np.asarray(calibration.get("P_front"), dtype=np.float64)
    P_side = np.asarray(calibration.get("P_side"), dtype=np.float64)
    if P_front.shape != (3, 4) or P_side.shape != (3, 4):
        raise HTTPException(status_code=500, detail="Stored projection matrices must be 3x4.")

    return calibration, P_front, P_side

@app.post("/upload")
async def upload_videos(
    front_video: UploadFile = File(...), 
    side_video: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())
    
    # Save files to disk/S3 here
    front_path = f"/tmp/{job_id}_front.mp4"
    side_path = f"/tmp/{job_id}_side.mp4"
    
    with open(front_path, "wb") as f:
        f.write(await front_video.read())
    with open(side_path, "wb") as f:
        f.write(await side_video.read())
        
    return {
        "status": "not_connected",
        "job_id": job_id,
        "message": (
            "Videos were received for preview storage only. User-uploaded 3D analysis "
            "requires calibrated camera projection matrices and is not connected yet."
        ),
        "analysis": None
    }


@app.post("/calibration/stereo")
async def calibrate_stereo_endpoint(
    front_images: List[UploadFile] = File(...),
    side_images: List[UploadFile] = File(...),
    board_rows: int = Form(...),
    board_cols: int = Form(...),
    square_size_mm: float = Form(...),
):
    if len(front_images) != len(side_images):
        raise HTTPException(status_code=422, detail="front_images and side_images counts must match.")

    with tempfile.TemporaryDirectory(prefix="golfpose_calibration_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        front_paths = []
        side_paths = []

        for index, upload in enumerate(front_images):
            suffix = Path(upload.filename or "").suffix or ".png"
            front_paths.append(await _save_upload(upload, temp_dir / f"front_{index}{suffix}"))
        for index, upload in enumerate(side_images):
            suffix = Path(upload.filename or "").suffix or ".png"
            side_paths.append(await _save_upload(upload, temp_dir / f"side_{index}{suffix}"))

        try:
            calibration = calibrate_stereo_cameras(
                front_paths,
                side_paths,
                board_rows,
                board_cols,
                square_size_mm,
            )
        except CalibrationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if calibration["valid_pair_count"] < 10:
        raise HTTPException(status_code=422, detail="At least 10 valid paired checkerboard detections are required.")
    if not np.isfinite(calibration["reprojection_error"]):
        raise HTTPException(status_code=422, detail="Stereo reprojection error is not finite.")
    if np.linalg.norm(np.asarray(calibration["T"], dtype=np.float64)) <= 1e-9:
        raise HTTPException(status_code=422, detail="Stereo baseline is too close to zero.")
    if np.asarray(calibration["P_front"]).shape != (3, 4) or np.asarray(calibration["P_side"]).shape != (3, 4):
        raise HTTPException(status_code=422, detail="Projection matrices must be 3x4.")

    calibration_payload = calibration_to_json(calibration, board_rows, board_cols, square_size_mm)
    save_calibration(calibration_payload, CALIBRATIONS_DIR)

    return {
        "status": "ok",
        "calibration_id": calibration_payload["calibration_id"],
        "valid_pair_count": calibration_payload["valid_pair_count"],
        "reprojection_error": calibration_payload["reprojection_error"],
        "K_front": calibration_payload["K_front"],
        "D_front": calibration_payload["D_front"],
        "K_side": calibration_payload["K_side"],
        "D_side": calibration_payload["D_side"],
        "R": calibration_payload["R"],
        "T": calibration_payload["T"],
        "P_front": calibration_payload["P_front"],
        "P_side": calibration_payload["P_side"],
        "calibration": {
            "available": True,
            "type": "stereo_checkerboard",
            "metric_scale": True,
            "notes": "Calibration generated from paired checkerboard images.",
        },
    }

@app.get("/demo/dlt-reconstruction")
async def get_demo_dlt_reconstruction():
    if not DEMO_FIXTURE_PATH.exists():
        raise HTTPException(status_code=500, detail="Calibrated demo fixture is missing.")

    try:
        with DEMO_FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)

        joint_names = fixture["joint_names"]
        projection_matrices = fixture["projection_matrices"]
        frames = reconstruct_dlt_frames(
            fixture["frames"],
            joint_names,
            projection_matrices["front"],
            projection_matrices["side"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Invalid calibrated demo fixture: {exc}") from exc

    return {
        "status": "ok",
        "source": fixture.get("source", "calibrated_demo"),
        "calibration": fixture.get("calibration", {
            "available": True,
            "type": "demo",
            "metric_scale": False,
            "notes": "Calibrated demo metadata was not provided."
        }),
        "joint_names": joint_names,
        "frame_count": len(frames),
        "frames": frames,
        "metrics": {
            "mpjpe_mm": None,
            "mpjpe_available": False,
            "notes": "MPJPE requires ground-truth 3D reference and is not computed for this demo."
        },
        "notes": fixture.get("notes", "")
    }


@app.post("/analysis/calibrated-dlt")
async def run_calibrated_dlt_analysis(
    front_video: UploadFile = File(...),
    side_video: UploadFile = File(...),
    calibration_id: str = Form(...),
):
    calibration, P_front, P_side = _load_calibration(calibration_id)
    job_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory(prefix="golfpose_analysis_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        front_suffix = Path(front_video.filename or "").suffix or ".mp4"
        side_suffix = Path(side_video.filename or "").suffix or ".mp4"
        front_path = await _save_upload(front_video, temp_dir / f"{job_id}_front{front_suffix}")
        side_path = await _save_upload(side_video, temp_dir / f"{job_id}_side{side_suffix}")

        try:
            front_keypoints = extract_pose_keypoints(front_path)
            side_keypoints = extract_pose_keypoints(side_path)
            sync_result = sync_two_view_keypoints(front_keypoints["frames"], side_keypoints["frames"])
            paired_frames = sync_result["paired_frames"]
            joint_names = [
                joint_name for joint_name in front_keypoints["joint_names"]
                if joint_name in set(side_keypoints["joint_names"])
            ]
            raw_frames3d = triangulate_sequence(paired_frames, joint_names, P_front, P_side)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Calibrated DLT analysis failed: {exc}") from exc

    smoothing_result = smooth_3d_frames(raw_frames3d)
    smoothed_frames3d = smoothing_result.pop("frames")
    analysis_frames3d = smoothed_frames3d if smoothing_result["available"] else raw_frames3d
    feature_frames = extract_biomechanical_features(analysis_frames3d)
    phase_features = extract_phase_features(feature_frames)
    phase_segmentation = segment_phases_with_svm(feature_frames)
    biomechanics = analyze_biomechanics(feature_frames, phase_segmentation)

    rendered_video = {
        "available": False,
        "path": None,
        "notes": "No triangulated 3D frames were available for rendering.",
    }
    if analysis_frames3d:
        output_path = RENDERED_DIR / f"{job_id}_3d_skeleton.mp4"
        try:
            rendered_path = render_3d_skeleton_video(analysis_frames3d, output_path)
            rendered_video = {
                "available": True,
                "path": rendered_path,
                "notes": "Rendered from triangulated 3D DLT frames.",
            }
        except (OSError, ValueError) as exc:
            rendered_video = {
                "available": False,
                "path": None,
                "notes": f"3D video rendering unavailable: {exc}",
            }

    return {
        "status": "ok",
        "source": "uploaded_calibrated_dlt",
        "job_id": job_id,
        "calibration_id": calibration["calibration_id"],
        "frame_count": len(analysis_frames3d),
        "joint_names": joint_names,
        "frames": analysis_frames3d,
        "raw_frames": raw_frames3d,
        "smoothed_frames": smoothed_frames3d,
        "smoothing": smoothing_result,
        "features": feature_frames,
        "biomechanics": biomechanics,
        "phase_features": phase_features,
        "phase_segmentation": phase_segmentation,
        "metrics": {
            "mpjpe_mm": None,
            "mpjpe_available": False,
            "notes": "MPJPE requires paired ground-truth 3D labels.",
        },
        "rendered_video": rendered_video,
    }


@app.get("/analysis/{job_id}/rendered-3d-video")
async def get_rendered_3d_video(job_id: str):
    output_path = RENDERED_DIR / f"{job_id}_3d_skeleton.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Rendered 3D video is not available for this job.")
    return {
        "status": "ok",
        "rendered_video": {
            "available": True,
            "path": str(output_path),
            "notes": "Rendered from triangulated 3D DLT frames.",
        },
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    # Fetch result from DB/Redis
    return {"job_id": job_id, "status": "Pending"}
