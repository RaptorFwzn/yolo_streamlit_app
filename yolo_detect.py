# yolo_detect.py
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def save_with_boxes(result, out_path: Path):
    """Render detections from an Ultralytics result onto the image (PIL) and save."""
    # Ultralytics gives BGR numpy array
    arr_bgr = result.orig_img
    arr_rgb = arr_bgr[..., ::-1]  # BGR -> RGB
    img = Image.fromarray(arr_rgb)
    draw = ImageDraw.Draw(img)

    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", None)

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            label_name = names[int(k)] if names and int(k) in names else str(k)
            label = f"{label_name} {c:.2f}"
            # box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            # text bg
            tw = draw.textlength(label, font=font)
            th = (font.size + 6) if font else 16
            x1t, y1t = x1, max(0, y1 - th)
            draw.rectangle([x1t, y1t, x1t + tw + 8, y1t + th], fill=(0, 255, 0))
            # text
            draw.text((x1t + 4, y1t + 3), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[INFO] Saved: {out_path}")

def run(model_path: str, source: str, save: bool = True, save_dir: str = "runs/streamlit"):
    if source.isdigit():
        print("[WARN] Webcam (numeric source) is not supported in cloud. Provide an image/video file path.")
        return

    model = YOLO(model_path)
    results = model(source)  # works for image or video; yields per-frame results

    save_root = Path(save_dir)
    idx = 0
    for r in results:
        if save:
            out_path = save_root / f"result_{idx}.jpg"
            save_with_boxes(r, out_path)
        idx += 1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to YOLO model (.pt or .onnx)")
    p.add_argument("--source", required=True, help="Path to image/video (not webcam index)")
    p.add_argument("--nosave", action="store_true", help="Do not save outputs")
    p.add_argument("--save_dir", default="runs/streamlit", help="Output folder")

    # Dummy args for compatibility with app.py / YOLO-style CLIs (ignored)
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
