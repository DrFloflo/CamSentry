# Jetson Orin Nano Super container setup

Target platform:

- JetPack 6.2.1
- Ubuntu 22.04.5 LTS
- Kernel 5.15.148-tegra
- L4T 36.4.7

## Files

- `Dockerfile.jetson`: Jetson/L4T image used to run WorldCam.
- `requirements-jetson.txt`: Python dependencies installed on top of the Jetson image. Do not install `torch` or `opencv-python` from PyPI here; use the Jetson image packages.
- `docker-compose.yaml`: Runs WorldCam in headless mode on port 8080.
- `.dockerignore`: Keeps virtualenvs and heavyweight model artifacts out of the Docker build context.

## Build

Run on the Jetson from the project root:

```bash
docker compose build worldcam
```

## Start

Foreground:

```bash
docker compose up worldcam
```

Background:

```bash
docker compose up -d worldcam
```

## Open the stream

From a browser on the same network:

```text
http://<jetson-ip>:8080/
```

Direct MJPEG stream:

```text
http://<jetson-ip>:8080/video_feed
```

## Logs

```bash
docker compose logs -f worldcam
```

## Stop

```bash
docker compose down
```

## GPU check

```bash
docker compose run --rm worldcam python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected result: `torch.cuda.is_available()` should print `True`.

## TensorRT engines

TensorRT `.engine` files are not portable across machines or TensorRT/CUDA versions.
Generate them directly on the Jetson, ideally inside this container, and store them in `models/`.

The compose file mounts local `./models` into `/app/models`, so generated engines persist outside the container.

## Performance notes

For an Orin Nano 8 Go, start with:

- headless mode enabled;
- segmentation disabled;
- pose disabled;
- SAHI disabled if FPS is too low;
- TensorRT engines generated locally on the Jetson;
- a smaller YOLO model if `yolo26l` is too slow or uses too much memory.

## Base image note

`Dockerfile.jetson` currently uses:

```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.6-py3
```

If NVIDIA changes available tags, replace it with an L4T R36 / JetPack 6 compatible PyTorch image for your installed L4T release.
