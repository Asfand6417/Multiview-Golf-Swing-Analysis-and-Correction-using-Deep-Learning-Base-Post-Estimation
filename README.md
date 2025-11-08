🏌️‍♂️ Multi-View Golf Swing Analysis System
3D Pose Estimation • Swing Phase Segmentation • Rule-Based Feedback

This project is a complete computer vision pipeline designed to analyze golf swings using multi-view synchronized cameras. It reconstructs 3D human poses, segments the golf swing into biomechanical phases, evaluates pose accuracy using MPJPE, and provides rule-based visual feedback to help improve swing performance.

🔍 Overview

The system processes dual-camera video input (side and back views) to detect 2D keypoints using OpenPose, reconstruct 3D joint coordinates via DLT (Direct Linear Transformation), and analyze the motion through SVM-based swing phase classification and rule-based evaluation.

Pipeline Summary:

1) Input: Two synchronized videos (Back View & Side View).

2) 2D Pose Detection: Keypoint extraction via Media Pipe.

3) 3D Pose Reconstruction: DLT triangulation from multi-view data.

4) Swing Phase Segmentation: Classification using SVM/MLP.

5) Pose Correction: Rule-based biomechanical evaluation.

6) Output: Annotated frames and videos with MPJPE, phase label, and feedback.

🧠 System Architecture



⚙️ Installation
1. Clone the Repository
git clone https://github.com/Asfand6417/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation.git
cd Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation

2. Set Up Environment
python -m venv venv
source venv/bin/activate     # (Linux/Mac)
venv\Scripts\activate        # (Windows)

3. Install Dependencies
pip install -r requirements.txt

▶️ Usage
1. Prepare Input Videos

Place synchronized back and side view videos inside data/videos/.

Ensure filenames match, e.g.

back_view/golfer1.mp4
side_view/golfer1.mp4

2. Run the Pipeline
python src/main.py --input golfer1


This will:

Detect 2D poses

Reconstruct 3D keypoints

Segment swing phases

Evaluate accuracy

Generate annotated outputs

3. View Results

Results are saved in:

output/annotated_frames/ – individual frames with pose overlays

output/result_videos/ – combined annotated swing video

output/metrics_report/ – MPJPE and evaluation report

📊 Evaluation Metrics
Metric	Description
MPJPE (Mean Per Joint Position Error)	Measures average 3D joint reconstruction accuracy.
Accuracy (%)	Phase classification accuracy using SVM.
Precision / Recall	Swing segmentation performance.
🎨 Visual Output Example
Correct Pose	Incorrect Pose

	

Annotated frames include:

Blue skeletons: Correct biomechanics

Red skeletons: Incorrect postures

Text overlay: Swing phase + feedback message

🧩 Core Technologies

OpenPose – 2D keypoint detection

DLT (Direct Linear Transformation) – 3D pose reconstruction

SVM / MLP – Swing phase classification

Rule-based engine – Posture correction feedback

OpenCV, NumPy, Scikit-learn, Matplotlib, Pandas

📚 Research Context

This project was developed as part of a Master’s thesis focused on golf swing biomechanics analysis using computer vision. It aims to support coaches and athletes in visualizing motion accuracy and improving swing performance through automated analysis.

🧾 Citation

If you use this project in your research or academic work, please cite:

Asfand Yar, "Multi-View Golf Swing Analysis using 3D Pose Estimation and Rule-Based Feedback", 2025.

💬 Contact

Author: Asfand Yar
Role: Web Designer • Computer Vision Expert • Researcher