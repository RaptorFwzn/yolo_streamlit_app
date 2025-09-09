import argparse
import cv2
from ultralytics import YOLO
import os

def run(model_path, source, show=True, save=True):
    # Load model
    model = YOLO(model_path)

    # Kalau source angka (misalnya "0") → gunakan webcam
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            print("Gagal membuka webcam!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Deteksi YOLO
            results = model(frame)
            for r in results:
                im_bgr = r.plot()

                if show:
                    cv2.imshow("YOLO Webcam Detection", im_bgr)

                if save:
                    out_dir = "runs/webcam_output"
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(out_dir, "last_frame.jpg"), im_bgr)

            # tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Kalau source file (gambar / video)
        results = model(source)
        for i, r in enumerate(results):
            im_bgr = r.plot()

            if show:
                cv2.imshow("YOLO Detection", im_bgr)
                cv2.waitKey(0)

            if save:
                out_dir = "runs/detect_output"
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"result_{i}.jpg")
                cv2.imwrite(out_path, im_bgr)
                print(f"Hasil disimpan di {out_path}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path ke model YOLO (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path ke gambar/video/webcam index (contoh: 0)")
    parser.add_argument("--noshow", action="store_true", help="Jangan tampilkan window hasil")
    parser.add_argument("--nosave", action="store_true", help="Jangan simpan hasil")
    args = parser.parse_args()

    run(
        model_path=args.model,
        source=args.source,
        show=not args.noshow,
        save=not args.nosave
    )
