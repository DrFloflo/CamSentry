# EarthCam stream documentation

This document explains how [`worldcam/main.py`](../main.py:1) opens and processes the EarthCam Dublin live stream.

## Purpose

[`worldcam/main.py`](../main.py:1) captures a protected EarthCam HLS stream, converts incoming video frames to grayscale with [`cv2.cvtColor()`](../main.py:144), and displays the result in an OpenCV window with [`cv2.imshow()`](../main.py:150).

The script exists because the EarthCam playlist URL requires browser-like HTTP headers. A plain request can be rejected even when the token is valid.

## Stream URL

The target stream is stored in [`stream_url`](../main.py:44). The current URL points to an EarthCam HLS playlist for the Dublin camera.

Important notes:

- The URL contains a token in the `t` query parameter.
- The URL contains a timestamp-like value in the `td` query parameter.
- The token may need to be refreshed from the EarthCam website when it expires.
- The website can play the stream while plain OpenCV access fails, because the browser sends headers such as `User-Agent`, `Referer`, and `Origin`.

## Dependencies

[`worldcam/main.py`](../main.py:1) uses:

- [`os`](../main.py:1) to set OpenCV FFmpeg environment options.
- [`shutil`](../main.py:2) to locate the external `ffmpeg` executable.
- [`subprocess`](../main.py:3) to start external FFmpeg as a frame pipe.
- [`urlparse()`](../main.py:4) to print stream host diagnostics.
- [`cv2`](../main.py:6) for OpenCV capture, conversion, display, and keyboard handling.
- [`numpy`](../main.py:7) to convert raw FFmpeg bytes into frame arrays.

External requirement:

- `ffmpeg` must be available in the system `PATH` for the fallback path used by [`start_ffmpeg_pipe()`](../main.py:74).

## Header configuration

[`configure_ffmpeg_http_headers()`](../main.py:10) configures OpenCV's bundled FFmpeg through the `OPENCV_FFMPEG_CAPTURE_OPTIONS` environment variable.

It sets:

- `user_agent` to mimic a desktop browser.
- `referer` to the EarthCam Dublin page.
- additional headers for `Origin` and `Accept`.

These headers are required because EarthCam can return `403 Forbidden` when the playlist is requested without browser-like headers.

## Diagnostics

[`print_videoio_diagnostics()`](../main.py:30) prints concise runtime information before opening the stream:

- OpenCV version from [`cv2.__version__`](../main.py:36).
- Stream host parsed by [`urlparse()`](../main.py:32).
- Whether OpenCV was built with FFmpeg support, checked through [`cv2.getBuildInformation()`](../main.py:33).
- Confirmation that EarthCam HTTP headers were enabled.

These diagnostics help distinguish between:

- a missing OpenCV FFmpeg backend,
- an expired or rejected EarthCam token,
- a protected stream that requires browser headers,
- OpenCV falling back to unrelated backends such as image sequence capture.

## Capture strategy

The script uses a two-step capture strategy.

### 1. Try OpenCV first

[`open_with_opencv()`](../main.py:62) tries to open [`stream_url`](../main.py:44) with [`cv2.VideoCapture()`](../main.py:64) and the [`cv2.CAP_FFMPEG`](../main.py:64) backend.

If OpenCV succeeds, frames are read directly with [`cap.read()`](../main.py:133).

If OpenCV fails, the script releases the capture object with [`cap.release()`](../main.py:69) and returns `None`, which triggers the external FFmpeg fallback.

### 2. Fallback to external FFmpeg

[`start_ffmpeg_pipe()`](../main.py:74) locates `ffmpeg` with [`shutil.which()`](../main.py:76). If it cannot find `ffmpeg`, it raises [`RuntimeError`](../main.py:78).

When `ffmpeg` is available, the function starts a subprocess with [`subprocess.Popen()`](../main.py:98). The command:

- sends browser-like headers using `-headers`,
- opens the HLS playlist with `-i`,
- disables audio with `-an`,
- scales output to `1280x720`,
- converts frames to BGR format with `-pix_fmt bgr24`,
- writes raw video frames to `pipe:1`.

This fallback works because external FFmpeg handles the protected HLS stream correctly with the required headers.

## Frame format

The fallback path uses fixed output dimensions:

- [`OUTPUT_WIDTH`](../main.py:46): `1280`
- [`OUTPUT_HEIGHT`](../main.py:47): `720`
- [`FRAME_SIZE`](../main.py:48): width multiplied by height multiplied by 3 BGR channels

[`read_ffmpeg_frame()`](../main.py:101) reads exactly one raw frame from the FFmpeg process. It expects [`FRAME_SIZE`](../main.py:48) bytes. If fewer bytes are received, frame reading fails and the main loop stops.

When a full frame is received, [`np.frombuffer()`](../main.py:110) converts the raw bytes into a NumPy array, then [`reshape()`](../main.py:110) converts it to `(720, 1280, 3)` BGR image shape.

## Main loop

[`main()`](../main.py:114) controls the complete program flow:

1. Calls [`configure_ffmpeg_http_headers()`](../main.py:115).
2. Prints diagnostics with [`print_videoio_diagnostics()`](../main.py:116).
3. Attempts direct OpenCV capture with [`open_with_opencv()`](../main.py:118).
4. Starts the external FFmpeg fallback with [`start_ffmpeg_pipe()`](../main.py:123) if OpenCV fails.
5. Reads frames from either [`cap.read()`](../main.py:133) or [`read_ffmpeg_frame()`](../main.py:135).
6. Converts each frame to grayscale with [`cv2.cvtColor()`](../main.py:144).
7. Displays the grayscale output with [`cv2.imshow()`](../main.py:150).
8. Exits when the user presses `q`, detected with [`cv2.waitKey()`](../main.py:153).

The script is protected by [`if __name__ == "__main__"`](../main.py:164), so importing [`worldcam/main.py`](../main.py:1) does not immediately start the camera loop.

## Cleanup

When the loop ends, [`main()`](../main.py:114) releases resources:

- Direct OpenCV capture is released with [`cap.release()`](../main.py:157).
- External FFmpeg is stopped with [`ffmpeg_process.terminate()`](../main.py:159).
- The script waits for FFmpeg shutdown with [`ffmpeg_process.wait()`](../main.py:160).
- OpenCV windows are closed with [`cv2.destroyAllWindows()`](../main.py:161).

## Troubleshooting

### OpenCV warning about `CAP_IMAGES`

If OpenCV prints an error mentioning `CAP_IMAGES`, it does not necessarily mean the URL is an image sequence. In this project, that warning is a fallback symptom after OpenCV fails to open the protected HTTPS HLS stream.

### Token is valid in the browser but OpenCV fails

This is expected for EarthCam streams. The browser sends required headers. [`configure_ffmpeg_http_headers()`](../main.py:10) attempts to configure those headers for OpenCV, and [`start_ffmpeg_pipe()`](../main.py:74) provides a reliable external FFmpeg fallback.

### `ffmpeg.exe` is missing

If [`shutil.which()`](../main.py:76) cannot find `ffmpeg`, [`start_ffmpeg_pipe()`](../main.py:74) raises [`RuntimeError`](../main.py:78). Install FFmpeg and ensure the executable is available in the system `PATH`.

### No frame received

If [`read_ffmpeg_frame()`](../main.py:101) cannot read [`FRAME_SIZE`](../main.py:48) bytes, the stream may have stopped, the token may have expired, or EarthCam may have changed its access rules.

## Extension point

The image-analysis area is inside the main loop after frame acquisition. Currently, frames are converted to grayscale with [`cv2.cvtColor()`](../main.py:144). This is where object detection, YOLO inference, motion detection, or recording logic can be added.
