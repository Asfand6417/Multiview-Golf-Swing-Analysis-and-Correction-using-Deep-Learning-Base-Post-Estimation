from celery import Celery

celery_app = Celery('analysis', broker='redis://redis:6379/0')

@celery_app.task
def process_golf_swing_videos(job_id: str, front_path: str, side_path: str):
    return {
        "job_id": job_id,
        "status": "not_connected",
        "message": (
            "Calibrated DLT analysis is disabled until Phase 2 wires a real "
            "pre-calibrated dataset into cv.reconstruction.reconstruct_3d."
        ),
        "analysis": None
    }
