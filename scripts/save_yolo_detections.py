import os
import os.path as osp
import pandas as pd
import math
import cv2
import numpy as np
from ultralytics import YOLO

# Paths to inputs and outputs
VIDEO_PATH = "/content/drive/MyDrive/Colab_Notebooks/Demos/Samurai/Raw_videos/7.mp4"  # Path to your input video
ALL_OBJECTS_TXT = "all_objects.txt"  # File to save all YOLO detections
TEMP_BBOX_TXT = "temp_bbox.txt"  # File to save the first frame of the top track
OUTPUT_VIDEO_PATH = "demo.mp4"  # File to save the SAMURAI output video
MODEL_PATH = "/content/drive/MyDrive/Colab_Notebooks/Data/YOLOv8/SportStory_v4/train/yolov8_custom/weights/best.pt" # Path to YOLO model weights
TRIMMED_VIDEO_PATH = "trimmed.mp4"

def run_yolo_with_motion_filter(video_path, all_objects_output, temp_bbox_output, model_path, conf_threshold=0.5):
    """
    Perform YOLO detection + tracking on a video and save detections to a file,
    filtering only moving soccer balls (class_id == 1) using optical flow.
    All other objects are tracked and saved without filtering.
    Save the first line of ball tracks to a separate file.
    """
    model = YOLO(model_path)
    prev_frame = None
    motion_threshold = 5  # Minimum movement in pixels to consider the ball moving
    frame_idx = 0
    ball_tracked = False
    start_frame = None

    with open(all_objects_output, 'w') as f:
        f.write("frame,class,track_id,x,y,w,h,speed\n")

    cap = cv2.VideoCapture(video_path)

    for result in model.track(
        source=video_path,
        conf=conf_threshold,
        stream=True,
        persist=True
    ):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue

        ret, current_frame = cap.read()
        if not ret:
            break

        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, gray_current,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

        with open(all_objects_output, 'a') as f:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                track_id = box.id[0] if box.id is not None else -1

                if cls == 1 and prev_frame is not None:
                    # Check if the ball has moved
                    region_flow = flow[y1:y2, x1:x2]
                    mean_flow = np.mean(np.sqrt(region_flow[..., 0]**2 + region_flow[..., 1]**2))

                    if mean_flow > motion_threshold:
                        f.write(f"{frame_idx},{cls},{track_id},{x1},{y1},{w},{h},{mean_flow:.2f}\n")
                        if not ball_tracked:
                            with open(temp_bbox_output, 'w') as temp_f:
                                temp_f.write(f"{x1},{y1},{w},{h}\n")
                            ball_tracked = True
                            start_frame = frame_idx
                elif cls != 1:
                    # Save all non-ball objects without motion filtering
                    f.write(f"{frame_idx},{cls},{track_id},{x1},{y1},{w},{h},0.00\n")

        prev_frame = gray_current
        frame_idx += 1

    cap.release()
    return start_frame

def trim_video_opencv(input_path, output_path, start_frame, end_frame):
    """
    Trim the video using OpenCV from the specified start frame to the end frame.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while cap.isOpened() and (end_frame is None or current_frame <= end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"Trimmed video saved to {output_path}")

# Execution
if __name__ == "__main__":
    # Run YOLO detection and tracking with motion filtering
    start_frame = run_yolo_with_motion_filter(
        video_path=VIDEO_PATH,
        all_objects_output=ALL_OBJECTS_TXT,
        temp_bbox_output=TEMP_BBOX_TXT,
        model_path=MODEL_PATH,
        conf_threshold=0.5
    )

    if start_frame is not None:
        # Trim video starting from the detected ball's first frame
        trim_video_opencv(
            input_path=VIDEO_PATH,
            output_path=TRIMMED_VIDEO_PATH,
            start_frame=start_frame,
            end_frame=None
        )
