<div align="center">

  <h1>Multi-View Golf Swing Analysis and Correction Using Deep Learning-Based Pose Estimation</h1>

  <p>
    An AI-assisted golf swing laboratory for multi-view 3D reconstruction,
    biomechanics analysis, swing-phase segmentation, and automated coaching feedback.
  </p>

  <p>
    <a href="https://github.com/Asfand6417/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation">GitHub Repository</a>
    |
    <a href="./docs/index.md">Documentation Index</a>
    |
    <a href="./docs/user_guide.md">User Guide</a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" />
    <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-Pose-4285F4?style=flat-square&logo=google&logoColor=white" />
    <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white" />
    <img alt="Scikit-Learn" src="https://img.shields.io/badge/Scikit--Learn-SVM-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
    <img alt="DLT" src="https://img.shields.io/badge/DLT-3D_Reconstruction-success?style=flat-square" />
  </p>

</div>

<img src="./images/Banner.png" alt="Golf Swing Biomechanics Intelligence" />

Overview
--------
Pro Swing AI (Golf) is a research-grade, product-oriented platform that converts synchronized multi-view swing video into accurate 3D biomechanics, actionable coaching feedback, and reproducible performance metrics. The product bridges laboratory-grade motion-capture capability and practical consumer camera workflows by combining robust computer vision, geometry, and machine learning.

Key capabilities
----------------
- Multi-view pose estimation and synchronized video ingestion
- Markerless motion capture using MediaPipe Pose
- DLT-based 3D triangulation and reconstruction
- Swing phase segmentation and kinematic feature extraction
- Rule-based fault detection with coaching suggestions
- MPJPE and related reconstruction quality reporting
- Interactive 3D visualizations and annotated playback
- Local-first design: video processing can run on-device or on-premises

Highlights
----------
| Capability | Description |
|------------|-------------|
| Multi-view biomechanics | Uses synchronized side and down-the-line camera views for robust 3D reconstruction |
| Markerless motion capture | Tracks motion using MediaPipe Pose without markers or wearables |
| DLT reconstruction | Triangulates 2D landmarks to produce accurate 3D joint coordinates |
| Swing phase segmentation | Automatically detects 8 swing phases for context-aware analysis |
| Biomechanics analysis | Computes X-Factor, spine angle, hip/shoulder rotation, wrist trajectory, and joint kinematics |
| Fault detection engine | Identifies common swing faults and provides corrective coaching feedback |
| Performance reporting | MPJPE reports, confusion matrices, and classification metrics |
| 3D visualization | Interactive skeletal playback and trajectory rendering |

Research results
----------------
| Metric | Result |
|--------|--------|
| Overall MPJPE | 8.77 mm |
| Setup MPJPE | 7.21 mm |
| Backswing MPJPE | 7.85 mm |
| Transition MPJPE | 9.53 mm |
| Downswing MPJPE | 10.76 mm |
| Impact MPJPE | 9.24 mm |
| Follow-through MPJPE | 8.03 mm |
| Expert Coaching Agreement | 85.3% |
| Processing Speed | 18.2 FPS |
| Pose Landmarks | 33 Keypoints |

System architecture
-------------------
<img src="./images/Diagram.jpeg" alt="System Architecture Diagram" />

Illustrations
-------------
<p align="center">
  <img src="./images/DLT 3D Tracking of Golfer's Swing.png" alt="DLT 3D Tracking of Golfer's Swing" style="max-width:100%;height:auto;border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,0.6)"/>
</p>
<p align="center"><em>DLT 3D Tracking of Golfer's Swing — triangulated joint cloud and reconstructed skeleton used for biomechanical feature extraction and visualization.</em></p>

<p align="center">
  <img src="./images/LANDMARK.png" alt="MediaPipe Landmark Topology" style="max-width:540px;height:auto;border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,0.4)"/>
</p>
<p align="center"><em>MediaPipe Pose landmark topology — 33 body keypoints used as the base 2D detections that feed DLT triangulation and biomechanical metrics.</em></p>

Flow (high level):
- Dual Camera Golf Swing Videos → Frame Synchronization → MediaPipe Pose Estimation → Keypoint Validation → DLT Triangulation → 3D Reconstruction → Trajectory Smoothing → Feature Engineering → Swing Phase Classification → Rule-Based Error Detection → Coaching Feedback → Annotated Reports & Visualizations

How it works
------------
<img src="./images/Pose estimation from Input video.svg" alt="Pose estimation from Input video" />

1. Multi-view video input (synchronized frames).
2. 2D pose detection per frame using MediaPipe; filtering by confidence.
3. DLT triangulation using calibrated camera matrices to recover 3D joint positions.
4. Smoothing (e.g., Savitzky–Golay) and temporal feature extraction.
5. Feature-based or learned phase segmentation (SVM / neural models).
6. Rule-based and learned fault detection (early extension, loss of spine angle, etc.).
7. Generate metrics, visual overlays, and coach-facing recommendations.

Swing phases
-------------

The system classifies each frame into one of eight phases:
1. Address
2. Takeaway
3. Backswing
4. Top
5. Transition
6. Downswing
7. Impact
8. Follow-through

Biomechanical metrics
---------------------
Measured and reported metrics include (non-exhaustive):
- MPJPE (3D reconstruction error)
- X-Factor (hip-to-shoulder separation)
- Hip / Shoulder rotation
- Spine angle and trunk mechanics
- Knee flexion and joint velocities
- Wrist trajectory and swing plane consistency
- Lead arm extension and impact geometry
- Performance and repeatability statistics

Fault detection
---------------
The expert-rule engine identifies common faults, including:
- Early Extension
- Over-the-Top swing path
- Loss of Spine Angle
- Insufficient Hip Rotation
- Reduced X-Factor
- Lead Arm Collapse
- Improper Knee Flexion
- Excessive Lateral Sway

<img src="./images/MJJPE.png" alt="MPJPE results visualization" />

Each detected fault includes an explanation and corrective recommendation.

Technology stack
----------------
| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Pose Estimation | MediaPipe Pose |
| Computer Vision | OpenCV |
| Numerical | NumPy, SciPy, Pandas |
| ML & DL | Scikit-Learn, PyTorch / TensorFlow optional |
| Reconstruction | Direct Linear Transformation (DLT) |
| Visualization | Matplotlib, Web-based 3D viewer |
| Backend API | FastAPI |
| Frontend | Next.js, TailwindCSS |
| Storage | PostgreSQL (optional) |
| Deployment | Docker, GPU-enabled inference nodes |

Project layout
--------------
```
Multiview-Golf-Swing-Analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── analysis/
│   │   ├── main.py
│   │   ├── PoseModule.py
│   │   ├── club_type_detector.py
│   │   ├── mediapipe_compat.py
│   │   └── test_triangulation.py
│   └── utils/
│       ├── video_processing.py
│       ├── display_info.py
│       ├── folder_management.py
│       ├── golf_rules.py
│       └── mpjpe_calculation.py
├── backend/
├── frontend/
├── data/
├── docs/
├── experiments/
├── assets/
├── images/
└── reports/
```

Installation
------------
1. Clone repository
```bash
git clone https://github.com/Asfand6417/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation.git
cd Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation
```

2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# or
.venv\Scripts\activate     # Windows
```

3. Install Python dependencies
```bash
pip install -r requirements.txt
```

4. Start API (example)
```bash
python src/analysis/main.py
```

5. Frontend
- Navigate to `frontend/` and run the Next.js app (`npm install` then `npm run dev`) — see frontend README for details.

Usage
-----
- Run the main analysis pipeline:
```bash
python src/analysis/main.py
```
- Validate triangulation:
```bash
python src/analysis/test_triangulation.py
```
- Run experimental multi-camera pipelines:
```bash
python experiments/multi_camera/approach_2/run_pipeline.py
```

Core modules
------------
| Module | Purpose |
|--------|---------|
| `src/analysis/main.py` | Main pipeline entry |
| `src/analysis/PoseModule.py` | MediaPipe wrapper and joint calc |
| `src/analysis/test_triangulation.py` | DLT validation |
| `src/analysis/club_type_detector.py` | Club classifier |
| `src/analysis/*` | Supporting modules for preprocessing, smoothing, and metrics |

Data & privacy
--------------
- Processing can run locally; athlete footage remains under user control.
- Store reports locally or on-premises Postgres.
- Do not publish footage without consent.

Roadmap
-------
- Real-time coaching and mobile support
- Transformer-based swing understanding
- Automated camera calibration
- Club and shaft-plane analysis
- Cloud analytics and benchmarking

Scientific citation
-------------------
```bibtex
@article{yar2026golfswing,
  title={Multi-View Golf Swing Analysis and Correction Using Markerless 3D Biomechanics},
  author={Yar, Asfand and others},
  journal={Scientific Reports},
  year={2026}
}
```

Acknowledgements
----------------
Built using MediaPipe, OpenCV, DLT, and modern sports biomechanics research to transform raw golf swing video into actionable coaching intelligence.



Git Troubleshooting
-------------------

### Common issues and solutions

- **Q: `ModuleNotFoundError` for installed packages?**
  - A: Ensure the virtual environment is activated. The terminal prompt should show `(.venv)` prefix. If not, activate it with:
    - `source .venv/bin/activate` (macOS/Linux)
    - `.venv\Scripts\activate` (Windows)

- **Q: `git pull` results in merge conflicts?**
  - A: Conflicts occur when local and remote branches have diverged. Resolve by editing the conflicted files to reconcile changes, then:
    ```bash
    git add <file-with-conflict>
    git commit
    ```

- **Q: `npm install` fails in the frontend directory?**
  - A: Ensure you have Node.js (v14+) and npm (v5.6+) installed. If issues persist, try deleting `node_modules` and `package-lock.json` then re-running `npm install`.

- **Q: Python dependency issues?**
  - A: Make sure to install all required packages in the virtual environment. If `pip install -r requirements.txt` fails, try installing dependencies individually or consult the package documentation for system requirements.

- **Q: Errors related to database connections?**
  - A: If using the optional PostgreSQL storage, ensure the database server is running and accessible. Check `backend/config.py` for connection settings.

- **Q: Issues with Docker deployment?**
  - A: Ensure Docker is installed and running. Check the Docker documentation for your OS if you encounter permission or service errors.

- **Q: Performance issues during video processing?**
  - A: Processing speed can vary based on system specifications and video length. For faster processing, use a machine with a dedicated GPU and sufficient RAM.

- **Q: How to reset changes and start over?**
  - A: To discard local changes and reset to the latest commit:
    ```bash
    git reset --hard HEAD
    git clean -fd
    ```
    **Warning: This will delete uncommitted changes and untracked files.**

Git Push Troubleshooting — non-fast-forward rejected
----------------------------------------------------

If `git push` fails with:

```
! [rejected]          main -> main (non-fast-forward)
error: failed to push some refs to 'https://...'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. If you want to integrate the remote changes,
hint: use 'git pull' before pushing again.
```

This means the remote branch contains commits your local branch does not. To resolve safely:

Quick one-off commands:
- Merge-style (safe, creates a merge commit):
  git pull --no-rebase origin main
  git push -u origin main

- Rebase-style (clean linear history; may require conflict resolution):
  git pull --rebase origin main
  git push -u origin main

- Fast-forward only (refuse non-fast-forward updates):
  git pull --ff-only origin main

Set a default globally:
- Prefer rebase:
  git config --global pull.rebase true
- Prefer merge:
  git config --global pull.rebase false
- Enforce fast-forward only:
  git config --global pull.ff only

If pull reports conflicts:
1. Resolve conflicts in indicated files.
2. git add <resolved-files>
3. If merging: git commit
   If rebasing: git rebase --continue
4. git push -u origin main

Helper script (recommended)
- A safe interactive helper is included: scripts/git-sync-and-push.sh
- Make it executable and run it from the repository root:
  chmod +x scripts/git-sync-and-push.sh
  ./scripts/git-sync-and-push.sh

This helper will:
- Fetch remote changes
- Attempt a fast-forward merge automatically
- If divergence exists, interactively offer Merge / Rebase / Inspect / Abort
- Perform the selected action and then push on success

Notes
- Avoid `git push --force` unless you intentionally want to overwrite remote history.
- For collaborative repos prefer merge or coordinated rebases to prevent disrupting others.