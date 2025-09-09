import streamlit as st
from pathlib import Path
import tempfile, time, sys, subprocess
from PIL import Image

# ---- Import cv2 aman (optional) ----
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

PROJECT_DIR = Path(__file__).resolve().parent
DETECT_SCRIPT = PROJECT_DIR / "yolo_detect.py"

# Folder hasil yang konsisten (penting di cloud)
RUNS_DIR = PROJECT_DIR / "runs" / "streamlit"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="YOLO Detector", page_icon="🟣", layout="centered")
st.title("🟣 YOLO Detector – Streamlit")

# ---------------- Sidebar ----------------
st.sidebar.header("Pengaturan")
DEFAULT_MODELS = [p for p in [PROJECT_DIR/"my_model.pt", PROJECT_DIR/"my_model.onnx"] if p.exists()]
model_choice = st.sidebar.selectbox(
    "Pilih model",
    options=([str(p) for p in DEFAULT_MODELS] + ["(Upload model lain...)"]) if DEFAULT_MODELS else ["(Upload model lain...)"],
    index=0 if DEFAULT_MODELS else 0,
)

if model_choice == "(Upload model lain...)":
    up_model = st.sidebar.file_uploader("Upload model (.pt / .onnx)", type=["pt","onnx"])
    if up_model:
        tmp_model = PROJECT_DIR / up_model.name
        with open(tmp_model, "wb") as f:
            f.write(up_model.read())
        model_path = str(tmp_model)
    else:
        model_path = ""
else:
    model_path = model_choice

# Di cloud tidak boleh menampilkan window GUI, jadi noshow=True by default
noshow = st.sidebar.checkbox("Gunakan --noshow (disarankan di cloud)", value=True)
# Agar hasil muncul, nosave=False
nosave = st.sidebar.checkbox("Gunakan --nosave (jangan simpan hasil)", value=False)

# ---------------- Input sumber ----------------
st.subheader("1) Pilih sumber")
tab_up, tab_cam = st.tabs(["📁 Upload Gambar", "📷 Kamera Browser"])

uploaded_file = None
with tab_up:
    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg","jpeg","png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Preview input", use_container_width=True)

camera_shot = None
with tab_cam:
    camera_shot = st.camera_input("Ambil foto dari kamera browser")
    if camera_shot:
        st.image(camera_shot, caption="Preview kamera browser", use_container_width=True)

# (Opsional) Tab webcam sistem akan disembunyikan saat di cloud:
if HAS_CV2:
    with st.expander("🎥 Webcam Sistem (OpenCV) — hanya untuk lokal, tidak didukung di cloud"):
        st.info("Fitur ini untuk menjalankan di komputer lokal.")
else:
    pass

# ---------------- Jalankan ----------------
st.subheader("2) Jalankan deteksi")
run_btn = st.button("▶️ Deteksi Sekarang")

def newest_output(after_ts: float):
    """Cari file gambar terbaru di RUNS_DIR setelah timestamp tertentu."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    newest, latest_ts = None, after_ts
    for p in RUNS_DIR.rglob("*"):
        if p.suffix.lower() in exts:
            ts = p.stat().st_mtime
            if ts >= latest_ts:
                newest, latest_ts = p, ts
    return newest

if run_btn:
    if not model_path:
        st.error("Model belum dipilih / diupload.")
    else:
        # Siapkan sumber
        source_path = None
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                source_path = Path(tmp.name)
        elif camera_shot:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(camera_shot.getvalue())
                source_path = Path(tmp.name)
        else:
            st.error("Belum ada sumber. Upload gambar atau ambil foto.")
        
        if source_path:
            st.info(f"Model: `{Path(model_path).name}`  |  Source: `{source_path.name}`")
            start_ts = time.time()

            # Build command
            cmd = [
                sys.executable, str(DETECT_SCRIPT),
                "--model", str(model_path),
                "--source", str(source_path),
                "--save_dir", str(RUNS_DIR),          # dipakai oleh yolo_detect.py revisi
            ]
            if noshow: cmd.append("--noshow")
            if nosave: cmd.append("--nosave")

            # Fallback untuk skrip yang gaya YOLO Ultralytics
            cmd += ["--project", str(RUNS_DIR.parent), "--name", RUNS_DIR.name, "--exist-ok"]

            with st.spinner("Inferensi berjalan..."):
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_DIR)

            with st.expander("Log deteksi"):
                st.code(proc.stdout or "(no stdout)")
                if proc.stderr:
                    st.code(proc.stderr)

            out_img = None if nosave else newest_output(after_ts=start_ts)
            if out_img:
                st.success(f"Hasil tersimpan: `{out_img}`")
                st.image(str(out_img), caption="Hasil deteksi", use_container_width=True)
                with open(out_img, "rb") as f:
                    st.download_button("⬇️ Download hasil", f, file_name=out_img.name)
            else:
                st.warning("Tidak menemukan file hasil (mungkin --nosave aktif). Menampilkan input.")
                st.image(Image.open(source_path).convert("RGB"), use_container_width=True)
