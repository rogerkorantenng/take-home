import cv2
import gradio as gr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle
import os

# Load YOLOv8 model
model = YOLO("yolov8l.pt")

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=50, embedder="mobilenet", embedder_gpu=True)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Error: Could not open video."

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    track_embeddings = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()

            if int(cls) == 0 and conf > 0.3:  # Person class
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append([bbox, conf, None])

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x, y, w, h = map(int, track.to_tlwh())

            embedding = track.features[-1] if track.features else None
            if embedding is not None:
                track_embeddings[track_id] = embedding.tolist()

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Ensure file is properly saved
    if not os.path.exists(output_path) or os.stat(output_path).st_size == 0:
        return "Error: Video file not generated correctly."

    with open("track_embeddings.pkl", "wb") as f:
        pickle.dump(track_embeddings, f)

    # Convert embeddings to a structured JSON format
    formatted_embeddings = "\n".join(
        [f"ID {track_id}: {embedding[:5]}..." for track_id, embedding in track_embeddings.items()]
    )

    return output_path, formatted_embeddings


def evaluate_model():
    """
    Evaluate the model using mAP and return the results.
    """
    results = model.val(data="coco128.yaml")

    map50 = results.box.map50  # mAP@0.5 (IoU threshold 0.5)
    map50_95 = results.box.map  # mAP averaged over IoU thresholds from 0.5 to 0.95

    return f"mAP@0.5: {map50:.4f}\nmAP@0.5:0.95: {map50_95:.4f}"


def gradio_process_video(video):
    output_path, formatted_embeddings = process_video(video)
    evaluation_result = evaluate_model()

    # ✅ Return file for download instead of displaying video
    return output_path, evaluation_result, formatted_embeddings


gr.Interface(
    fn=gradio_process_video,
    inputs=gr.Video(),
    outputs=[
        gr.File(label="Download Processed Video"),  # ✅ Now a downloadable file
        gr.Textbox(label="mAP Evaluation"),
        gr.Textbox(label="Track Embeddings (Matched with ID)", interactive=True, lines=10)
    ],
    title="Person Detection & Tracking with mAP Evaluation",
    description="Upload a video to detect and track people. Evaluates detection accuracy using mAP and displays embeddings linked to track IDs. Download the processed video after analysis.",
).launch(share=True)
