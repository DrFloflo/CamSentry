# Cam-Detect: YOLOv11 RTSP Stream Processor

Cam-Detect is a Python-based application designed to monitor and process multiple RTSP camera streams in real-time. It leverages the power of a YOLOv11 model to perform object detection, identifies specific classes (e.g., "person", "cat"), and triggers a webhook with an annotated image upon detection.

## Features

- **Multi-Stream Processing:** Concurrently handles multiple camera channels using threading.
- **Real-time Object Detection:** Utilizes a YOLOv11 model for fast and accurate detections.
- **Webhook Notifications:** Sends a POST request to a specified URL with a base64-encoded image of the detection event.
- **Configurable:** Easily configured through environment variables (`.env` file).
- **Performance Tuned:** Includes frame skipping and adjustable inference width to balance performance and resource usage.
- **Graceful Shutdown:** Captures system signals (SIGINT, SIGTERM) to stop the application cleanly.
- **Development Mode:** Provides a local preview of the camera streams with detection overlays for debugging and development.
- **GPU Acceleration:** Automatically uses CUDA for inference if a compatible GPU is available, with a fallback to CPU.

## Prerequisites

- Python 3.8+
- `pip` for package management
- A compatible YOLOv11 model file (e.g., `yolo11l.pt`)
- Pytorch for you specific CUDA version

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Cam-detect
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download YOLOv11 Model:**
    Place your trained YOLOv11 model file (e.g., `yolo11l.pt`) in the root directory of the project.

## Configuration

Configuration is managed via a `.env` file in the root of the project. Create this file by copying the example:

```bash
cp .env.example .env
```

Then, edit the `.env` file with your specific settings:

| Variable          | Description                                                                                             | Example                               |
| ----------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `RTSP_URL_BASE`   | The base URL for your RTSP streams. The channel number is appended automatically.                         | `rtsp://user:pass@192.168.1.100:554/` |
| `CAMERA_CHANNELS` | A comma-separated list of camera channel numbers to process.                                              | `1,3,4`                               |
| `WEBHOOK_URL`     | The URL to which the detection notification (POST request with JSON payload) will be sent.                | `http://localhost:8080/webhook`       |
| `INFERENCE_WIDTH` | The width (in pixels) to which frames are resized before inference. Higher values may improve accuracy but decrease performance. | `640`                                 |
| `FRAME_SKIP`      | Process every Nth frame to save resources. A value of `1` processes every frame.                          | `2`                                   |
| `ENVIRONMENT`     | Set to `development` to enable local video preview windows, or `production` to run headless.              | `development`                         |

## Usage

To start the application, run the main script from the root directory:

```bash
python main.py
```

The application will start processing the camera channels specified in your `.env` file.

## Graceful Shutdown

To stop the application safely, press `Ctrl+C` in the terminal where it is running. The application will catch the signal, stop all processing threads, and clean up resources before exiting.