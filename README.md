# Multiview Golf Swing Analysis and Correction

3D Pose Estimation • Swing Phase Segmentation • Rule-Based Feedback

[Live GitHub Page](https://asfand6417.github.io/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation/) •
[Documentation](docs/index.md) •
[Issues](https://github.com/Asfand6417/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation/issues)

## Project Goals

- Deliver accurate 3D pose reconstruction from multiple camera views.
- Segment golf swings into meaningful phases for coaching analysis.
- Provide explainable, rule-based feedback to improve swing mechanics.
- Offer a research-ready toolkit for biomechanics and sports analytics.

## Key Features

- **Multi-camera fusion** for stable 3D pose reconstruction.
- **Swing phase classification** (Address → Follow-through).
- **Quantitative metrics** such as MPJPE, joint angles, and smoothness.
- **Interactive 3D visualizations** for coaching and research review.
- **Modular pipeline** with reusable scripts and documentation.

## Core Technologies

- **Python** for processing and orchestration
- **MediaPipe** for 2D pose detection
- **OpenCV** for video processing
- **NumPy / Pandas** for data wrangling
- **scikit-learn** for swing phase classification (SVM / MLP)
- **TensorFlow** for deep learning components
- **Matplotlib** for visualizations

## Repository Map

- `main.py` – Core pipeline for 3D pose analysis and visualization.
- `PoseModule.py` – MediaPipe wrapper for pose detection.
- `Multi-Camera Approach-1/` – Multi-view calibration and triangulation pipeline.
- `docs/` – Project documentation and GitHub Pages site.
- `results/` – Generated outputs and reports.

## Installation

### Prerequisites

- Python 3.8+
- A GPU is recommended for deep learning workflows (optional).

### Setup

```bash
git clone https://github.com/Asfand6417/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation.git
cd Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage Examples

### Single-camera analysis

```python
from main import Pose3DVisualizer

visualizer = Pose3DVisualizer()
visualizer.process_video("PoseVideos/golf_swing.mp4")
visualizer.animate(save=True, output_path="results/analyzed_swing.mp4")
```

### Multi-camera reconstruction

```bash
python "Multi-Camera Approach-1/main_pipeline.py"
# Update input paths inside the script to point at your synchronized camera footage.
```

### Quick validation scripts

```bash
python test_triangulation.py
```

## Demo Video & Visuals

- Add a short demo clip to `docs/assets/demo.mp4` and the GitHub Pages site will display it.
- Optional: link a YouTube/Vimeo demo in the GitHub Pages “Demo” section.

## Documentation

The documentation includes setup guides, pipeline explanations, and module references:

- [Documentation Index](docs/index.md)
- [User Guide](docs/user_guide.md)
- [Main Module](docs/main_module.md)
- [Multi-Camera System](docs/multicamera_system.md)

## GitHub Pages Deployment

The website is built from the `docs/` folder.

1. Go to **Settings → Pages** in the GitHub repository.
2. Set **Source** to **Deploy from a branch**.
3. Select **Branch: main** and **Folder: /docs**.
4. Save and wait for the deployment to complete.

The site will be available at:
`https://asfand6417.github.io/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation/`

## Research Context

This project was developed as part of a Master’s thesis focused on golf swing biomechanics analysis using computer vision. It aims to support coaches and athletes in visualizing motion accuracy and improving swing performance through automated analysis.

## Citation

If you use this project in your research or academic work, please cite:

Asfand Yar, "Multi-View Golf Swing Analysis using 3D Pose Estimation and Rule-Based Feedback", 2025.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
