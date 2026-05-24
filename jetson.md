# Jetson Orin Nano Super container setup

Target platform:

- JetPack 6.2.1
- Ubuntu 22.04.5 LTS
- Kernel 5.15.148-tegra
- L4T 36.4.7

## Files

- `docker-compose.yaml`: Runs WorldCam directly from the official Ultralytics Jetson image in headless mode on port 8080.
- `Dockerfile.jetson`: kept only as an experimental/custom build reference; it is not used by the current compose service.
- `requirements-jetson.txt`: kept only for custom builds; it is not installed by the current compose service.
- `.dockerignore`: used only when building a custom image.

## Pull the official Ultralytics Jetson image

Run on the Jetson:

```bash
docker pull ultralytics/ultralytics:latest-jetson-jetpack6
```

The compose service uses this image directly and does not build a custom image.

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

## Image note

`docker-compose.yaml` uses:

```yaml
image: ultralytics/ultralytics:latest-jetson-jetpack6
```

This avoids custom pip installation of Ultralytics, PyTorch and CUDA packages. If this image does not contain one of the extra app dependencies, install it only after checking that pip does not replace `torch`, `torchvision`, `opencv-python` or CUDA packages.

## Quick manual run without compose

From the project root on the Jetson:

```bash
docker run --rm -it \
  --ipc=host \
  --runtime=nvidia \
  --network=host \
  -w /app \
  -v "$PWD:/app" \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  python3 -m worldcam.main --headless --stream-host 0.0.0.0 --stream-port 8080
```
