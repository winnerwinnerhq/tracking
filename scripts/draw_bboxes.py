import cv2

def draw_bounding_boxes(video_path, annotations_path, output_path):
    # Read the annotations from all_objects.txt
    annotations = {}
    with open(annotations_path, 'r') as f:
        for line in f.readlines()[1:]:  # Skip the header
            frame, cls, track_id, x, y, w, h, speed = line.strip().split(',')
            frame = int(frame)
            x, y, w, h = int(x), int(y), int(w), int(h)
            if frame not in annotations:
                annotations[frame] = []
            annotations[frame].append((cls, track_id, x, y, w, h))

    # Open the original video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in annotations:
            for cls, track_id, x, y, w, h in annotations[frame_idx]:
                color = (0, 255, 0) if cls == '0' else (255, 0, 0)  # Green for ball, blue for others
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")

if __name__ == "__main__":
    video_path = "/content/drive/MyDrive/Colab_Notebooks/Demos/Samurai/Raw_videos/7.mp4"
    annotations_path = "all_objects.txt"
    output_path = "annotated_video.mp4"
    draw_bounding_boxes(video_path, annotations_path, output_path)