# yolo_detect.py — gunakan renderer Ultralytics (tanpa OpenCV)
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np

def run(model_path: str, source: str, save: bool = True, save_dir: str = "runs/streamlit"):
    # Webcam (angka) tidak didukung di cloud
    if source.isdigit():
        print("[WARN] Source numerik (webcam) tidak didukung di cloud. Berikan path file.")
        return

    model = YOLO(model_path)
    results = model(source)  # bekerja untuk gambar / video; mengembalikan per-frame Results

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results):
        # Pakai renderer internal Ultralytics -> BGR numpy array dengan kotak+label sudah tergambar
        im_bgr = r.plot()  # <- INI kuncinya

        # Konversi BGR -> RGB tanpa cv2, simpan via PIL
        im_rgb = im_bgr[..., ::-1]  # BGR->RGB
        img = Image.fromarray(im_rgb.astype(np.uint8))

        out_path = out_dir / f"result_{i}.jpg"
        if save:
            img.save(out_path)
            print(f"[INFO] Saved: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path YOLO model (.pt/.onnx)")
    p.add_argument("--source", required=True, help="Path gambar/video (bukan index webcam)")
    p.add_argument("--nosave", action="store_true", help="Jangan simpan output")
    p.add_argument("--save_dir", default="runs/streamlit", help="Folder output")

    # Dummy args biar kompatibel dengan app.py/YOLO CLI (diabaikan)
    p.add_argument("--noshow", action="store_true")
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--exist-ok", action="store_true")

    args = p.parse_args()
    run(
        model_path=args.model,
        source=args.source,
        save=not args.nosave,
        save_dir=args.save_dir
    )
