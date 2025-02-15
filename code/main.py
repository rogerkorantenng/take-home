import os
import cv2
import gradio as gr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle

# Ensure 'model' folder exists
os.makedirs("model", exist_ok=True)

# Define model path
MODEL_PATH = "model/yolov8l.pt"

# Download YOLO model if not found
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model...")
    model = YOLO("yolov8l.pt")  # Load model
    model.export(format="torchscript", save_dir="model")  # Save in 'model' folder
else:
    model = YOLO(MODEL_PATH)  # Load model from 'model' folder

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=50, embedder="mobilenet", embedder_gpu=True)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Error: Could not open video."

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Output video file
    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Dictionary to store embeddings and IDs
    track_embeddings = {}

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
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert (x1, y1, x2, y2) → (x, y, w, h)
                detections.append([bbox, conf, None])  # None is placeholder for embedding

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes and IDs
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x, y, w, h = map(int, track.to_tlwh())  # Convert tracking bbox to correct format

            # Get embedding
            embedding = track.features[-1] if track.features else None  # Get the latest embedding
            if embedding is not None:
                track_embeddings[track_id] = embedding.tolist()  # Save embedding as list

            # Draw bounding box and ID label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Save embeddings and IDs
    with open("track_embeddings.pkl", "wb") as f:
        pickle.dump(track_embeddings, f)

    return output_path

def evaluate_model():
    """
    Evaluate the model using mAP and return the results.
    """
    results = model.val(data="coco128.yaml")  # Use a dataset like COCO128 or your custom dataset

    map50 = results.box.map50  # mAP@0.5 (IoU threshold 0.5)
    map50_95 = results.box.map  # mAP averaged over IoU thresholds from 0.5 to 0.95

    return f"mAP@0.5: {map50:.4f}\nmAP@0.5:0.95: {map50_95:.4f}"

# Gradio UI
def gradio_process_video(video):
    output_path = process_video(video)
    evaluation_result = evaluate_model()
    return output_path, evaluation_result

# Create interactive Gradio app
gr.Interface(
    fn=gradio_process_video,
    inputs=gr.Video(),
    outputs=[gr.Video(), gr.Textbox(label="mAP Evaluation")],
    title="Person Detection & Tracking with mAP Evaluation",
    description="Upload a video to detect and track people. Evaluates detection accuracy using mAP."
).launch(share=True)
