import argparse
import os
import os.path as osp
import pandas as pd
import math
import cv2
import numpy as np
import torch
import gc
import sys
from ultralytics import YOLO

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

# Paths to inputs and outputs
VIDEO_PATH = "/content/drive/MyDrive/Colab_Notebooks/Demos/Samurai/Raw_videos/7.mp4"  # Path to your input video
ALL_OBJECTS_TXT = "all_objects.txt"  # File to save all YOLO detections
TEMP_BBOX_TXT = "temp_bbox.txt"  # File to save the first frame of the top track
OUTPUT_VIDEO_PATH = "demo.mp4"  # File to save the SAMURAI output video
MODEL_PATH = "/content/drive/MyDrive/Colab_Notebooks/Data/YOLOv8/SportStory_v4/train/yolov8_custom/weights/best.pt" # Path to YOLO model weights
TRIMMED_VIDEO_PATH = "trimmed.mp4"

color = [(255, 0, 0)]

def run_yolo_with_motion_filter(video_path, all_objects_output, temp_bbox_output, model_path, conf_threshold=0.5):
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
                    f.write(f"{frame_idx},{cls},{track_id},{x1},{y1},{w},{h},0.00\n")

        prev_frame = gray_current
        frame_idx += 1

    cap.release()
    return start_frame

def trim_video_opencv(input_path, output_path, start_frame, end_frame):
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

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def save_ball_tracks(tracks_output_file, frame_idx, bbox, speed):
    with open(tracks_output_file, 'a') as f:
        x, y, w, h = bbox
        f.write(f"{frame_idx},1,0,{x},{y},{w},{h},{speed:.2f}\n")

def main(args):
    start_frame = run_yolo_with_motion_filter(
        video_path=args.video_path,
        all_objects_output=ALL_OBJECTS_TXT,
        temp_bbox_output=TEMP_BBOX_TXT,
        model_path=args.yolo_model_path,
        conf_threshold=0.5
    )

    if start_frame is not None:
        trim_video_opencv(
            input_path=args.video_path,
            output_path=TRIMMED_VIDEO_PATH,
            start_frame=start_frame,
            end_frame=None
        )

    model_cfg = determine_model_cfg(args.samurai_model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.samurai_model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(TRIMMED_VIDEO_PATH)
    prompts = load_txt(TEMP_BBOX_TXT)

    frame_rate = 30
    if args.save_to_video:
        if osp.isdir(TRIMMED_VIDEO_PATH):
            frames = sorted([osp.join(TRIMMED_VIDEO_PATH, f) for f in os.listdir(TRIMMED_VIDEO_PATH) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(TRIMMED_VIDEO_PATH)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    tracks_output_file = "ball_tracks.txt"
    with open(tracks_output_file, 'w') as f:
        f.write("frame,class,track_id,x,y,w,h,speed\n")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

                if obj_id == 0:
                    x, y, w, h = bbox
                    speed = 0.0
                    save_ball_tracks(tracks_output_file, frame_idx, (x, y, w, h), speed)

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                out.write(img)

        if args.save_to_video:
            out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default=VIDEO_PATH, help="Input video path or directory of frames.")
    parser.add_argument("--yolo_model_path", default=MODEL_PATH, help="Path to YOLO model weights.")
    parser.add_argument("--samurai_model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the SAMURAI model checkpoint.")
    parser.add_argument("--video_output_path", default=OUTPUT_VIDEO_PATH, help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    args = parser.parse_args()
    main(args)