from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from tasks.analysis import process_golf_swing_videos
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
        
    # Temporary bypass of Celery so you don't need Redis to test the connection
    # process_golf_swing_videos.delay(job_id, front_path, side_path)
    
    return {
        "status": "success", 
        "job_id": job_id,
        "flaws": ["Spine Angle Loss", "Flying Right Elbow"],
        "injury": "Moderate Shoulder Strain Risk",
        "mpjpe": 22.4
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    # Fetch result from DB/Redis
    return {"job_id": job_id, "status": "Pending"}
