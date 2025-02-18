import cv2
import gradio as gr
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=50, embedder="mobilenet", embedder_gpu=True)

# Function to calculate mAP (Mean Average Precision) safely
def evaluate_map():
    try:
        metrics = model.val(split="val")
        map_50 = metrics.box.map50
        map_50_95 = metrics.box.map
        return map_50, map_50_95
    except Exception as e:
        print(f"Error during mAP evaluation: {e}")
        return "N/A", "N/A"

# Process video function
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Error: Could not open video.", None, None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Output video file
    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO on the frame
        results = model(frame)[0]

        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()

            # Only detect people (class 0)
            if int(cls) == 0 and conf > 0.3:
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append([bbox, conf, None])

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes and IDs
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x, y, w, h = map(int, track.to_tlwh())

            # Draw bounding box and ID label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    map_50, map_50_95 = evaluate_map()

    return output_path, f"mAP@0.5: {map_50}", f"mAP@0.5:0.95: {map_50_95}"

def gradio_process_video(video):
    output_path, map_50, map_50_95 = process_video(video)
    return output_path, map_50, map_50_95

gr.Interface(
    fn=gradio_process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Video(label="Processed Video"),
        gr.Textbox(label="mAP@0.5 Score"),
        gr.Textbox(label="mAP@0.5:0.95 Score")
    ],
    title="Person Detection & Tracking",
    description="Upload a video to detect and track people in real-time. The processed video and accuracy metrics will be displayed."
).launch()
