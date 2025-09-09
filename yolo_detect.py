# yolo_detect.py — PIL only, kotak biru tebal + debug
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

BLUE  = (0, 180, 255)     # warna kotak & label
WHITE = (255, 255, 255)   # warna teks

def draw_rect_thick(draw: ImageDraw.ImageDraw, xy, color, width=4):
    """Gambar rectangle tebal dengan 4 garis, supaya pasti terlihat di semua versi PIL."""
    x1, y1, x2, y2 = map(int, xy)
    for w in range(width):
        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=color)

def save_with_boxes(result, out_path: Path):
    """Render bbox + label (tanpa OpenCV) lalu simpan."""
    arr_bgr = result.orig_img
    if arr_bgr is None:
        print("[WARN] result.orig_img is None — skip.")
        return

    # Ultralytics memberi BGR -> ubah ke RGB tanpa cv2
    arr_rgb = arr_bgr[..., ::-1]
    img = Image.fromarray(arr_rgb)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", None)

    if boxes is None or len(boxes) == 0:
        print("[INFO] Tidak ada bbox pada frame ini.")
    else:
        xyxy = boxes.xyxy.cpu().numpy().astype(int)   # (N, 4)
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        print(f"[INFO] Jumlah bbox: {len(xyxy)}")
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            # clamp agar tetap di kanvas
            x1 = max(0, min(int(x1), W - 1))
            y1 = max(0, min(int(y1), H - 1))
            x2 = max(0, min(int(x2), W - 1))
            y2 = max(0, min(int(y2), H - 1))
            if x2 <= x1 or y2 <= y1:
                print(f"[WARN] Bbox invalid: {(x1,y1,x2,y2)} — dilewati.")
                continue

            label_name = names[int(k)] if (names and int(k) in names) else str(k)
            label = f"{label_name} {c:.2f}"
            print(f"[BOX] {label}: {(x1,y1,x2,y2)}")

            # kotak tebal (lebih “nendang” daripada width=1)
            draw_rect_thick(draw, (x1, y1, x2, y2), BLUE, width=4)

            # ukuran label
            if hasattr(draw, "textbbox"):
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            else:
                tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else len(label) * 7
                th = (font.size + 6) if font else 16

            # posisi label (di atas kotak; kalau mepet, taruh di dalam)
            ly1 = y1 - th - 4
            if ly1 < 0:
                ly1 = y1 + 4
            draw.rectangle([x1, ly1, x1 + tw + 10, ly1 + th + 6], fill=BLUE)
            draw.text((x1 + 5, ly1 + 3), label, fill=WHITE, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[INFO] Saved: {out_path}")

def run(model_path: str, source: str, save: bool = True, save_dir: str = "runs/streamlit"):
    # Webcam (angka) tidak didukung di cloud
    if source.isdigit():
        print("[WARN] Source numerik (webcam) tidak didukung di cloud. Berikan path gambar/video.")
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
    # dummy args agar kompatibel dengan app.py
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
