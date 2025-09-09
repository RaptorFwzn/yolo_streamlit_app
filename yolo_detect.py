import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def save_with_boxes(result, out_path):
    arr_bgr = result.orig_img  # numpy array BGR
    arr_rgb = arr_bgr[..., ::-1]  # BGR -> RGB
    img = Image.fromarray(arr_rgb)
    draw = ImageDraw.Draw(img)

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)
        try:
            font = ImageFont.load_default()
        except:
            font = None

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            label = f"{result.names[int(k)]} {c:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=3)
            draw.text((x1+3, y1+3), label, fill=(0,0,0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[INFO] Hasil disimpan di {out_path}")

def run(model_path, source, save=True, save_dir="runs/streamlit"):
    model = YOLO(model_path)
    results = model(source)
    save_dir = Path(save_dir)
    for i, r in enumerate(results):
        if save:
            save_with_boxes(r, save_dir / f"result_{i}.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--save_dir", type=str, default="runs/streamlit")
    args = parser.parse_args()

    run(
        model_path=args.model,
        source=args.source,
        save=not args.nosave,
        save_dir=args.save_dir
    )
