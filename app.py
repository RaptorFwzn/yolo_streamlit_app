import argparse
import cv2
from ultralytics import YOLO
import os
from pathlib import Path

def run(model_path, source, show=True, save=True, save_dir="runs/streamlit"):
    # Load model
    model = YOLO(model_path)

    # Pastikan folder output ada
    save_dir = Path(save_dir)
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Kalau source angka (misalnya "0") → gunakan webcam
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            print("Gagal membuka webcam!")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Deteksi YOLO
            results = model(frame)
            for r in results:
                im_bgr = r.plot()

                if show:
                    # ⚠️ imshow tidak jalan di cloud, jadi hanya untuk lokal
                    cv2.imshow("YOLO Webcam Detection", im_bgr)

                if save:
                    out_path = save_dir / f"webcam_frame_{frame_count}.jpg"
                    cv2.imwrite(str(out_path), im_bgr)
                    print(f"[INFO] Hasil disimpan di {out_path}")
                    frame_count += 1

            # tekan 'q' untuk keluar (hanya berfungsi di lokal dengan jendela imshow)
            if show and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Kalau source file (gambar / video)
        results = model(source)
        for i, r in enumerate(results):
            im_bgr = r.plot()

            if show:
                # ⚠️ imshow hanya untuk lokal
                cv2.imshow("YOLO Detection", im_bgr)
                cv2.waitKey(0)

            if save:
                out_path = save_dir / f"result_{i}.jpg"
                cv2.imwrite(str(out_path), im_bgr)
                print(f"[INFO] Hasil disimpan di {out_path}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path ke model YOLO (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path ke gambar/video/webcam index (contoh: 0)")
    parser.add_argument("--noshow", action="store_true", help="Jangan tampilkan window hasil")
    parser.add_argument("--nosave", action="store_true", help="Jangan simpan hasil")
    parser.add_argument("--save_dir", type=str, default="runs/streamlit", help="Folder untuk menyimpan hasil")  # 👈 tambahan
    args = parser.parse_args()

    run(
        model_path=args.model,
        source=args.source,
        show=not args.noshow,
        save=not args.nosave,
        save_dir=args.save_dir
    )
