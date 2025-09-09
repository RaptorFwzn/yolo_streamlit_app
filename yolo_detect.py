import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_boxes_pil(result, out_path):
    """
    Gambar bounding boxes dari 'result' (Ultralytics) ke gambar,
    lalu simpan ke 'out_path' tanpa menggunakan OpenCV.
    """
    # result.orig_img = numpy array BGR; ubah ke RGB tanpa cv2
    arr_bgr = result.orig_img
    arr_rgb = arr_bgr[..., ::-1]  # BGR -> RGB
    img = Image.fromarray(arr_rgb)
    draw = ImageDraw.Draw(img)

    # Ambil prediksi
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()        # (N, 4)
        conf = boxes.conf.cpu().numpy()        # (N,)
        cls  = boxes.cls.cpu().numpy().astype(int)  # (N,)

        # Font opsional (safe default)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            label = f"{result.names[int(k)]} {c:.2f}" if hasattr(result, "names") else f"id{k} {c:.2f}"
            # kotak
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            # latar label sederhana
            tw, th = draw.textlength(label, font=font), 12 if font is None else font.size + 4
            draw.rectangle([x1, max(0, y1 - th), x1 + tw + 6, y1], fill=(0, 255, 0))
            draw.text((x1 + 3, y1 - th + 2), label, fill=(0, 0, 0), font=font)

    # Simpan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[INFO] Hasil disimpan di {out_path}")

def run(model_path, source, show=True, save=True, save_dir="runs/streamlit"):
    """
    show: diabaikan (tidak ada GUI di cloud), disediakan agar kompatibel.
    """
    # Tolak webcam di cloud
    if str(source).isdigit():
        print("[WARN] Mode webcam tidak didukung di lingkungan cloud.")
        return

    # Load model
    model = YOLO(model_path)

    # Jalankan inferensi pada file gambar / video
    # (Untuk video, Ultralytics akan mengeluarkan banyak 'result')
    results = model(source)

    save_dir = Path(save_dir)
    idx = 0
    for r in results:
        out_path = save_dir / f"result_{idx}.jpg"
        if save:
            draw_boxes_pil(r, out_path)
        idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path ke model YOLO (.pt / .onnx)")
    parser.add_argument("--source", type=str, required=True, help="Path ke gambar/video (bukan index webcam)")
    parser.add_argument("--noshow", action="store_true", help="(Diabaikan) Jangan tampilkan window hasil")
    parser.add_argument("--nosave", action="store_true", help="Jangan simpan hasil")
    parser.add_argument("--save_dir", type=str, default="runs/streamlit", help="Folder untuk menyimpan hasil")
    # argumen kompatibilitas dari app.py (tidak dipakai tapi tidak bikin error)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    run(
        model_path=args.model,
        source=args.source,
        show=not args.noshow,
        save=not args.nosave,
        save_dir=args.save_dir
    )
