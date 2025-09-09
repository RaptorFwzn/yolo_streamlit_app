# yolo_detect.py (versi PIL-only, kotak biru + label biru)
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

BLUE = (0, 180, 255)     # biru kotak & label
WHITE = (255, 255, 255)  # teks

def save_with_boxes(result, out_path: Path):
    """Render bbox + label (tanpa OpenCV) lalu simpan."""
    # result.orig_img adalah numpy BGR
    arr_bgr = result.orig_img
    if arr_bgr is None:
        return
    arr_rgb = arr_bgr[..., ::-1]  # BGR -> RGB
    img = Image.fromarray(arr_rgb)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", None)

    if boxes is not None and len(boxes) > 0:
        # ambil ndarray dan pastikan int
        xyxy = boxes.xyxy.cpu().numpy().astype(int)     # (N, 4)
        conf = boxes.conf.cpu().numpy()                 # (N,)
        cls  = boxes.cls.cpu().numpy().astype(int)      # (N,)

        # font aman (default PIL)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            # clamp biar tidak keluar kanvas
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W - 1))
            y2 = max(0, min(y2, H - 1))
            if x2 <= x1 or y2 <= y1:
                continue  # skip kotak buruk

            # nama kelas
            label_name = names[int(k)] if (names and int(k) in names) else str(k)
            label = f"{label_name} {c:.2f}"

            # gambar kotak
            draw.rectangle([x1, y1, x2, y2], outline=BLUE, width=3)

            # ukuran teks (pakai textbbox biar akurat)
            if hasattr(draw, "textbbox"):
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            else:
                # fallback lama
                tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else len(label) * 7
                th = (font.size + 6) if font else 16

            # posisi label: di atas kotak, kalau mepet atas, taruh di dalam kotak
            ly1 = y1 - th - 2
            if ly1 < 0:
                ly1 = y1 + 2
            # latar label
            draw.rectangle([x1, ly1, x1 + tw + 8, ly1 + th + 4], fill=BLUE)
            # teks
            draw.text((x1 + 4, ly1 + 2), label, fill=WHITE, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[INFO] Saved: {out_path}")

def run(model_path: str, source: str, save: bool = True, save_dir: str = "runs/streamlit"):
    if source.isdigit():
        print("[WARN] Webcam (angka) tidak didukung di cloud. Gunakan file gambar/video.")
        return

    model = YOLO(model_path)
    results = model(source)

    save_root = Path(save_dir)
    for i, r in enumerate(results):
        if save:
            out_path = save_root / f"result_{i}.jpg"
            save_with_boxes(r, out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path model YOLO (.pt / .onnx)")
    p.add_argument("--source", required=True, help="Path gambar/video (bukan index webcam)")
    p.add_argument("--nosave", action="store_true", help="Jangan simpan hasil")
    p.add_argument("--save_dir", default="runs/streamlit", help="Folder output")

    # argumen dummy agar kompatibel dengan app.py/YOLO CLI (diabaikan)
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
