
<div align="center">

  <h1>Golf Swing Biomechanics Intelligence</h1>

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

## Overview

Golf Swing Biomechanics Intelligence is a research-driven computer vision platform for automated golf swing analysis and correction.

The system combines:

- Multi-camera pose estimation
- Direct Linear Transformation (DLT)
- 3D biomechanical reconstruction
- Swing phase classification
- Rule-based fault detection
- Coach-oriented feedback generation

Unlike conventional monocular systems, the framework reconstructs true three-dimensional motion from synchronized camera views, enabling accurate measurement of rotational mechanics, swing plane characteristics, X-Factor separation, spine posture, impact dynamics, and sequencing efficiency.

The platform was developed to bridge the gap between expensive laboratory motion-capture systems and affordable consumer-camera-based golf coaching solutions.

---

## Highlights

| Capability | Description |
|------------|-------------|
| Multi-view biomechanics | Uses synchronized side-view and down-the-line cameras for robust 3D reconstruction |
| Markerless motion capture | Tracks golfer movement using MediaPipe Pose without wearable sensors |
| DLT reconstruction | Generates accurate 3D joint coordinates through triangulation |
| Swing phase segmentation | Detects Address, Takeaway, Backswing, Top, Transition, Downswing, Impact, and Follow-through |
| Biomechanical analysis | Computes X-Factor, spine angle, hip rotation, shoulder rotation, wrist trajectory, and joint kinematics |
| Fault detection engine | Identifies common swing faults using biomechanics rules |
| Coaching recommendations | Converts kinematic measurements into actionable corrective feedback |
| Performance evaluation | Generates MPJPE reports, confusion matrices, and classification metrics |
| 3D visualization | Interactive skeletal playback and trajectory rendering |

---

<img src="https://github.com/Asfand6417/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation/blob/4ed18370ae828b5d1995257ea3a61e565d0b83b6/MJJPE.png" />


## Research Results

| Metric | Result |
|----------|---------|
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

---

## System Architecture
<img src="./images/Diagram.jpeg" alt="Golf Swing Biomechanics Intelligence" />

flowchart LR

A["Dual Camera Golf Swing Videos"]
--> B["Frame Synchronization"]

B --> C["MediaPipe Pose Estimation"]

C --> D["Keypoint Validation"]

D --> E["DLT Triangulation"]

E --> F["3D Reconstruction"]

F --> G["Trajectory Smoothing"]

G --> H["Feature Engineering"]

H --> I["Swing Phase Classification"]

H --> J["Biomechanics Analysis"]

I --> K["Rule-Based Error Detection"]

J --> K

K --> L["Coaching Feedback"]

L --> M["Annotated Reports & Visualizations"]


---

## How It Works
<img src="./images/Diagram.jpeg" alt="Golf Swing Biomechanics Intelligence" />
flowchart LR
    A["Multi-View Swing Video"]
    --> B["Pose Detection & Landmark Extraction"]

   B --> C["Confidence Filtering"]

   C --> D["DLT Triangulation"]

   D --> E["3D Pose Reconstruction"]

   E --> F["Savitzky-Golay Smoothing"]

   F --> G["Biomechanics Feature Extraction"]

   G --> H["SVM Swing Phase Segmentation"]

   H --> I["Rule-Based Fault Detection"]

   I --> J["Performance Metrics"]

   J --> K["Coaching Feedback"]


---

## Swing Phases
<img src="./images/Diagram.jpeg" alt="Golf Swing Biomechanics Intelligence" />
The system automatically classifies each frame into one of eight golf swing phases:

1. Address
2. Takeaway
3. Backswing
4. Top
5. Transition
6. Downswing
7. Impact
8. Follow-through

These phases form the foundation for biomechanical analysis and context-aware coaching recommendations.

---

## Biomechanical Metrics

| Metric             | Purpose                                         |
| ------------------ | ----------------------------------------------- |
| MPJPE              | Quantifies 3D reconstruction accuracy           |
| X-Factor           | Measures hip-to-shoulder rotational separation  |
| Hip Rotation       | Evaluates rotational power generation           |
| Shoulder Rotation  | Assesses upper-body sequencing                  |
| Spine Angle        | Monitors posture retention throughout the swing |
| Lead Arm Extension | Evaluates backswing structure                   |
| Wrist Trajectory   | Tracks club delivery path                       |
| Swing Plane        | Measures movement consistency                   |
| Knee Flexion       | Assesses setup and impact stability             |
| Joint Velocities   | Measures kinetic chain sequencing               |

---

## Fault Detection

The expert-rule engine identifies common golf swing errors including:

* Early Extension
* Over-the-Top Swing Path
* Loss of Spine Angle
* Insufficient Hip Rotation
* Reduced X-Factor
* Lead Arm Collapse
* Improper Knee Flexion
* Excessive Lateral Sway

Each detected issue is accompanied by corrective recommendations and biomechanical explanations.

---

## Technology Stack

| Layer                | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python 3.10                  |
| Pose Estimation      | MediaPipe Pose               |
| Computer Vision      | OpenCV                       |
| Numerical Processing | NumPy, Pandas, SciPy         |
| Machine Learning     | Scikit-Learn                 |
| Classification       | SVM (RBF Kernel)             |
| Reconstruction       | Direct Linear Transformation |
| Visualization        | Matplotlib                   |
| Deep Learning        | TensorFlow (Optional)        |

---

## Project Structure


Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation/

├── README.md
├── requirements.txt
├── main.py
├── PoseModule.py
├── club_type_detector.py
├── mediapipe_compat.py
├── test_triangulation.py

├── docs/
│   ├── index.md
│   ├── user_guide.md
│   ├── main_module.md
│   ├── pose_detection.md
│   └── multicamera_system.md

├── Multi-Camera Approach-1/
│   └── main_pipeline.py

├── Multi-Camera Approach-2/
│   └── run_pipeline.py

├── GolfDB/

├── images/
│   ├── Banner.png
│   ├── Diagram.jpeg
│   ├── Graphs/
│   ├── results/
│   ├── Multi-Camera Approach-1/
│   ├── Multi-Camera Approach-2/
│   └── best classifier/


---

## Installation

### Clone Repository

 git clone https://github.com/Asfand6417/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation.git

cd Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation


### Create Environment


python -m venv .venv


#### Linux / macOS


source .venv/bin/activate


#### Windows


.venv\Scripts\activate


### Install Dependencies


pip install -r requirements.txt


---

## Usage

### Run Main Analysis Pipeline

```bash
python main.py
```

### Run Triangulation Validation

```bash
python test_triangulation.py
```

### Run Multi-Camera Pipeline

```bash
python "Multi-Camera Approach-2/run_pipeline.py"
```

---

## Core Modules

| File                      | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `main.py`                 | Main golf swing analysis pipeline             |
| `PoseModule.py`           | MediaPipe pose wrapper and joint calculations |
| `club_type_detector.py`   | Driver vs Iron classification                 |
| `test_triangulation.py`   | Multi-camera reconstruction validation        |
| `Multi-Camera Approach-1` | Initial reconstruction experiments            |
| `Multi-Camera Approach-2` | Production multi-view pipeline                |
| `GolfDB`                  | Dataset and annotations                       |
| `Graphs`                  | Evaluation visualizations                     |
| `results`                 | Generated outputs and reports                 |

---

## Data And Privacy

* All video processing can run locally.
* No cloud service is required.
* Athlete footage remains under user control.
* Generated reports and metrics are stored locally.
* Avoid publishing athlete recordings without consent.

---

## Future Roadmap

* Real-time coaching feedback
* Transformer-based swing understanding
* Automated camera calibration
* Club tracking and shaft-plane analysis
* Mobile deployment
* Cloud analytics dashboard
* Professional swing comparison engine
* Multi-athlete benchmarking

---

## Scientific Contributions

This project integrates:

* Markerless Motion Capture
* Multi-View Geometry
* 3D Human Pose Reconstruction
* Sports Biomechanics
* Machine Learning Classification
* Expert-System Coaching Feedback

to provide an accessible and scalable golf swing analysis framework.

---

## Citation

```bibtex
@article{yar2026golfswing,
  title={Multi-View Golf Swing Analysis and Correction Using Markerless 3D Biomechanics},
  author={Yar, Asfand and others},
  journal={Scientific Reports},
  year={2026}
}
```

---

## Acknowledgements

Built using MediaPipe Pose, OpenCV, Direct Linear Transformation (DLT), Multi-View Geometry, Machine Learning, and modern sports biomechanics research to transform raw golf swing videos into actionable coaching intelligence.

