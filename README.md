# Soccer Video Analysis Pipeline

This project processes amateur soccer videos to detect and track players, goalposts, and the ball, while also identifying key soccer events and creating highlight clips. The pipeline leverages state-of-the-art computer vision techniques and includes several optimized steps for video analysis.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Pipeline Steps](#pipeline-steps)
5. [Output](#output)

---

## **Overview**

This project analyzes amateur soccer videos recorded on smartphones. The pipeline detects players, goalposts, and the ball, tracks their movement, identifies key soccer events, and produces highlight videos. It is optimized for low-resolution and dynamic videos typical of Sunday league matches.

---

## **Features**

- Player and goalpost detection using YOLOv8.
- Dynamic ball tracking using Optical Flow and SAMURAI.
- Key event detection (e.g., goals, assists) with precise timestamps.
- Automated video trimming and highlight extraction.
- Visualized bounding boxes overlaid on the original video.

---

## **Requirements**

- Python 3.8+
- Dependencies:
  - OpenCV
  - YOLOv8
  - DeepSORT
  - SAMURAI (for ball tracking)
  - Optical Flow (built-in with OpenCV)
  - Additional Python libraries: `numpy`, `pandas`, `matplotlib`

---


## **Pipeline Steps**

1. **Object Detection:**
   - Process the video using YOLOv8 to detect players and goalposts.

2. **Player Tracking:**
   - Use DeepSORT to track players and goalposts across frames.

3. **Save Detections:**
   - Store detections in a structured TXT file.

4. **Dynamic Ball Tracking:**
   - Detect the ball using Optical Flow and SAMURAI, saving coordinates and frame numbers.

5. **Video Trimming:**
   - Trim the video from the first detected frame of the ball.

6. **Event Prediction:**
   - Analyze tracking data to identify goals, assists, and other key events.

7. **Highlight Extraction:**
   - Extract highlight clips based on detected event timestamps.

8. **Visualization:**
   - Overlay bounding boxes and annotations onto the original video.

---

## **Output**

- Annotated video with bounding boxes and labels.
- TXT files containing detection and tracking data.
- Highlight clips for key soccer events.

