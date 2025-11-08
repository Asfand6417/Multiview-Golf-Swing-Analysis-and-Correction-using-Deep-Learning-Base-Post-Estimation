# Pose Detection Module Documentation

The Pose Detection module (`PoseModule.py`) provides a simplified interface to MediaPipe's pose detection capabilities, making it easy to detect and analyze human poses in images and videos.

## Overview

The `poseDetector` class encapsulates MediaPipe's pose detection functionality and adds useful methods for finding poses, extracting landmark positions, and calculating angles between body parts. This module is particularly useful for 2D pose analysis before moving to 3D reconstruction.

## Class: poseDetector

### Initialization

```python
def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5)
```

Parameters:
- `mode` (bool): If set to True, the detector will detect poses in each image. If False, it will also track poses across frames for better performance.
- `smooth` (bool): Whether to apply landmark smoothing.
- `detectionCon` (float): Minimum confidence value (0.0-1.0) for pose detection to be considered successful.
- `trackCon` (float): Minimum confidence value (0.0-1.0) for pose landmarks to be considered tracked successfully.

### Methods

#### findPose

```python
def findPose(self, img, draw=True)
```

Detects pose in an image and optionally draws the landmarks and connections.

Parameters:
- `img` (numpy.ndarray): Input image (BGR format from OpenCV).
- `draw` (bool): Whether to draw landmarks and connections on the image.

Returns:
- The image with landmarks and connections drawn (if `draw=True`).

#### findPosition

```python
def findPosition(self, img, draw=True)
```

Extracts the pixel coordinates of all pose landmarks.

Parameters:
- `img` (numpy.ndarray): Input image.
- `draw` (bool): Whether to draw circles at landmark positions.

Returns:
- A list of landmark positions, where each element is [id, x, y] (id is the landmark index, x and y are pixel coordinates).

#### findAngle

```python
def findAngle(self, img, p1, p2, p3, draw=True)
```

Calculates the angle between three points (landmarks).

Parameters:
- `img` (numpy.ndarray): Input image.
- `p1`, `p2`, `p3` (int): Indices of three landmarks forming an angle, with p2 as the vertex.
- `draw` (bool): Whether to visualize the angle on the image.

Returns:
- The angle in degrees.

## MediaPipe Pose Landmarks

The module uses MediaPipe's pose landmarks, which include 33 key points on the human body:

0. Nose
1-4. Left and right eye (inner, outer)
5-8. Left and right ear
9-10. Mouth (left, right)
11-12. Left and right shoulders
13-14. Left and right elbows
15-16. Left and right wrists
17-22. Left and right hands (pinky, index, thumb)
23-24. Left and right hips
25-26. Left and right knees
27-28. Left and right ankles
29-32. Left and right feet (heel, toe)

## Usage Examples

### Basic Pose Detection

```python
import cv2
from PoseModule import poseDetector

# Initialize detector
detector = poseDetector()

# Read an image or video frame
img = cv2.imread("path/to/image.jpg")

# Find pose
img = detector.findPose(img)

# Get landmark positions
landmarks = detector.findPosition(img)

# Display the image
cv2.imshow("Pose Detection", img)
cv2.waitKey(0)
```

### Calculating Joint Angles

```python
import cv2
from PoseModule import poseDetector

# Initialize detector
detector = poseDetector()

# Read an image or video frame
img = cv2.imread("path/to/image.jpg")

# Find pose
img = detector.findPose(img)

# Get landmark positions
landmarks = detector.findPosition(img)

if len(landmarks) > 0:
    # Calculate angle between right shoulder (12), elbow (14), and wrist (16)
    angle = detector.findAngle(img, 12, 14, 16)
    print(f"Right arm angle: {angle} degrees")
    
    # Calculate angle between left shoulder (11), elbow (13), and wrist (15)
    angle = detector.findAngle(img, 11, 13, 15)
    print(f"Left arm angle: {angle} degrees")

# Display the image
cv2.imshow("Joint Angles", img)
cv2.waitKey(0)
```

### Processing Video

```python
import cv2
from PoseModule import poseDetector

# Initialize detector
detector = poseDetector()

# Open video file or camera
cap = cv2.VideoCapture("path/to/video.mp4")  # or 0 for webcam

while True:
    success, img = cap.read()
    if not success:
        break
        
    # Find pose
    img = detector.findPose(img)
    
    # Get landmark positions
    landmarks = detector.findPosition(img)
    
    if len(landmarks) > 0:
        # Example: Track right elbow position
        if 14 < len(landmarks):
            elbow_x, elbow_y = landmarks[14][1], landmarks[14][2]
            cv2.circle(img, (elbow_x, elbow_y), 15, (0, 0, 255), cv2.FILLED)
    
    # Display the image
    cv2.imshow("Video Pose Detection", img)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Integration with GolfPose

The PoseModule is used in the GolfPose system for initial 2D pose detection before more advanced 3D analysis. It provides the foundation for:

1. Detecting the golfer's pose in each frame
2. Extracting landmark positions for further analysis
3. Calculating joint angles for swing analysis
4. Providing visual feedback on the detected pose

## Best Practices

For optimal results with the Pose Detection module:

1. **Lighting and Visibility**:
   - Ensure good lighting conditions
   - Make sure the subject is fully visible in the frame
   - Avoid loose or baggy clothing that can obscure body contours

2. **Camera Setup**:
   - Position the camera at an appropriate distance to capture the full body
   - For golf swing analysis, a side view is often most informative
   - Ensure the camera is stable (use a tripod if possible)

3. **Performance Optimization**:
   - Set `mode=False` for video processing to enable tracking
   - Adjust `detectionCon` and `trackCon` based on your needs:
     - Higher values for more precision but potentially fewer detections
     - Lower values for more detections but potentially less precision

4. **Angle Calculations**:
   - When using `findAngle()`, ensure the three points form a meaningful angle
   - For golf swing analysis, common angles to track include:
     - Shoulder-elbow-wrist angles for arm position
     - Hip-knee-ankle angles for stance
     - Shoulder-hip-knee angles for posture