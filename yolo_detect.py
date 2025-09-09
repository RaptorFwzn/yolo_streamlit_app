# yolo_detect.py — PIL only, kotak biru + fallback xyxyn
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

BLUE  = (0, 180, 255)     # warna kotak & label (biru)
WHITE = (255, 255, 255)   # warna teks

def _clamp(v, lo, hi):
    return max(lo, min(int(v), hi))

def _text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    # fallback
    w = draw.textlength(text, font=font) if hasattr(draw, "textlength") else len(text) * 7
    h = (font.size + 6) if font else 16
    return int(w), int(h)

def _draw_box(draw, x1, y1, x2, y2, label, font):
    # kotak tebal
    for w in range(4):
        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=BLUE)
    # label di atas kotak (atau di dalam jika mepet tepi)
    tw, th = _text_size(draw, label, font)
    ly1 = y1 - th - 4
    if ly1 < 0:
        ly1 = y1 + 4
    draw.rectangle([x1, ly1, x1 + tw + 10, ly1 + th + 6], fill=BLUE)
    draw.text((x1 + 5, ly1 + 3), label, fill=WHITE, font=font)

def save_with_boxes(result, out_path: Path):
    """Render bbox + label (tanpa OpenCV) lalu simpan."""
    arr_bgr = result.orig_img
    if arr_bgr is None:
        print("[WARN] result.orig_img is None — skip.")
        return

    # BGR -> RGB (tanpa cv2)
    img = Image.fromarray(arr_bgr[..., ::-1])
    draw = ImageDraw.Draw(img)
    W, H = img.size

    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", None)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if boxes is None or len(boxes) == 0:
        print("[INFO] Tidak ada bbox pada frame ini.")
    else:
        # 1) coba xyxy (piksel). 2) jika invalid, pakai xyxyn (ter-normalisasi)
        xyxy = getattr(boxes, "xyxy", None)
        fallback_to_norm = False
        if xyxy is None or len(xyxy) == 0:
            fallback_to_norm = True
        else:
            xyxy = xyxy.cpu().numpy()
            if not np.any(xyxy):  # semua nol?
                fallback_to_norm = True

        if fallback_to_norm:
            xyxyn = boxes.xyxyn.cpu().numpy()  # [0..1]
            print("[INFO] Fall back ke xyxyn (normalized boxes).")
            xyxy = np.zeros_like(xyxyn)
            xyxy[:, 0] = xyxyn[:, 0] * W
            xyxy[:, 1] = xyxyn[:, 1] * H
            xyxy[:, 2] = xyxyn[:, 2] * W
            xyxy[:, 3] = xyxyn[:, 3] * H

        conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy))
        cls  = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy), int)

        print(f"[INFO] Jumlah bbox (after fallback): {len(xyxy)}")
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            x1 = _clamp(x1, 0, W - 1); y1 = _clamp(y1, 0, H - 1)
            x2 = _clamp(x2, 0, W - 1); y2 = _clamp(y2, 0, H - 1)
            if x2 <= x1 or y2 <= y1:
                print(f"[WARN] Bbox invalid: {(x1,y1,x2,y2)} — dilewati.")
                continue

            name = names[int(k)] if (names and int(k) in names) else str(k)
            label = f"{name} {float(c):.2f}"
            print(f"[BOX] {label}: {(x1,y1,x2,y2)}")
            _draw_box(draw, x1, y1, x2, y2, label, font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[INFO] Saved: {out_path}")

def run(model_path: str, source: str, save: bool = True, save_dir: str = "runs/streamlit"):
    if source.isdigit():
        print("[WARN] Source numerik (webcam) tidak didukung di cloud. Gunakan path file.")
        return

    model = YOLO(model_path)  # ONNX / PT keduanya OK
    results = model(source)

    save_root = Path(save_dir)
    for i, r in enumerate(results):
        if save:
            out_path = save_root / f"result_{i}.jpg"
            save_with_boxes(r, out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path YOLO model (.pt/.onnx)")
    p.add_argument("--source", required=True, help="Path gambar/video (bukan index webcam)")
    p.add_argument("--nosave", action="store_true")
    p.add_argument("--save_dir", default="runs/streamlit")
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
