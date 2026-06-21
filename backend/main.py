import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid

try:
    from cv.reconstruction import reconstruct_dlt_frames
except ModuleNotFoundError:
    from backend.cv.reconstruction import reconstruct_dlt_frames

app = FastAPI(title="GolfPose API")
DEMO_FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "calibrated_dlt_demo.json"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    # Fetch result from DB/Redis
    return {"job_id": job_id, "status": "Pending"}
