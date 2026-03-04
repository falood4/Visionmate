import os
import cv2
import subprocess
import numpy as np
from ultralytics import YOLO

os.environ["QT_QPA_PLATFORM"] = "xcb"

# ── Set up the camera with rpicam-vid ─────────
WIDTH  = 1280
HEIGHT = 1280
FPS    = 15

cmd = [
    "rpicam-vid",
    "--width",     str(WIDTH),
    "--height",    str(HEIGHT),
    "--framerate", str(FPS),
    "--codec",     "yuv420",
    "--output",    "-",
    "--nopreview",
    "--timeout",   "0",
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
frame_bytes = WIDTH * HEIGHT * 3 // 2


def capture_frame():
    """Read one frame from rpicam-vid pipe and return as BGR."""
    data = b""
    while len(data) < frame_bytes:
        chunk = proc.stdout.read(frame_bytes - len(data))
        if not chunk:
            return None
        data += chunk
    yuv = np.frombuffer(data, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


# ── Load YOLOv8 NCNN model ────────────────────
model = YOLO("yolov8n_ncnn_model")

while True:
    # Capture a frame from the camera
    frame = capture_frame()
    if frame is None:
        break

    # Run YOLO model on the captured frame and store the results
    results = model(frame)

    # Output the visual detection data, draw on camera preview window
    annotated_frame = results[0].plot()

    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert from milliseconds
    text = f'FPS: {fps:.1f}'

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10                              # 10 pixels from the top

    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Close all windows
proc.terminate()
proc.wait()
cv2.destroyAllWindows()