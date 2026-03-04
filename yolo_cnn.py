from ultralytics import YOLO

PT_MODEL   = "yolov8n.pt"
INFER_SIZE = 320

print(f"[INFO] Loading {PT_MODEL} ...")
model = YOLO(PT_MODEL)

print(f"[INFO] Exporting to NCNN format (imgsz={INFER_SIZE}) ...")
print("[INFO] This will take 1-2 minutes, please wait ...")
model.export(format="ncnn", imgsz=INFER_SIZE)

print("[INFO] Done! NCNN model saved to: yolov8n_ncnn_model/")
print("[INFO] You can now run: python3 yolo_live.py")
