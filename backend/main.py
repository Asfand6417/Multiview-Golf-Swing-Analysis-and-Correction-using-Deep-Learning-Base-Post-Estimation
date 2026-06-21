from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI(title="GolfPose API")

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

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    # Fetch result from DB/Redis
    return {"job_id": job_id, "status": "Pending"}
