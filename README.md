# Person Detection & Tracking with YOLOv8 and DeepSORT

## Summary

This project implements a real-time person detection and tracking system using **YOLOv8** and **DeepSORT**. The system processes video input to detect and track people, assigning unique IDs to each individual and displaying bounding boxes around them. Additionally, the model evaluates its detection performance using the **mean Average Precision (mAP)** metric, providing insights into the accuracy of the detection.

Key features include:
- **YOLOv8** for detecting people (class 0) in video frames.
- **DeepSORT** for tracking people across frames and assigning unique IDs.
- Evaluation of the detection performance using **mAP** at different IoU thresholds (0.5 and 0.5:0.95).
- **Gradio** interface to allow users to upload videos, view processed results, and see mAP evaluation results.
- Tracking embeddings are saved for further analysis.

---

## Prerequisites & Running Application

Before running the system, ensure you have the following installed:

### 1. Python 3.8+ 
   - Make sure Python 3.8 or a later version is installed on your system. You can download it from the official [Python website](https://www.python.org/downloads/).

### 2. Required Libraries
   - The following Python libraries are required for the project **cv2**, **gradio**, **ultralytics**, **DeepSort** . You can install them using `pip`:

### 3. Using the requirements.txt file to install all libraries

```bash
   pip3 install -r requirements.txt
   ```

### 4. How to Run the Application
To run the application run 
```bash
   python3 code/main.py
   ```

## System Tested On

The system has been tested on the following environments:

### Operating System:
   - Ubuntu 24.04 LTS

### Python Version:
   - Python 3.8+

### Libraries:
   - OpenCV version 4.5.3
   - Gradio version 3.0
   - Ultralytics YOLOv8 version 8.0
   - DeepSORT-Realtime version 1.0

### Hardware:
   - **CPU**: Intel Ultra 7
   - **GPU**: NVIDIA RTX 5000 Ada
   - **RAM**: 32 GB
