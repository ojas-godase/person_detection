import streamlit as st
import tempfile, os, sys, subprocess, cv2, numpy as np, torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------- SETTINGS ----------------
MODEL_PATH = "yolov5_person_best.pt"  # your YOLOv5 weights
st.set_page_config(page_title="People Counting (YOLOv5)", layout="wide")
st.title("👥 People Detection & Counting App — YOLOv5")

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
iou_thresh  = st.sidebar.slider("IoU Threshold",        0.1, 1.0, 0.45, 0.05)

# NEW: control YOLOv5 inference size
imgsz = st.sidebar.select_slider(
    "Inference Size (YOLOv5 imgsz)",
    options=[640, 736, 832, 960, 1088, 1216, 1344, 1536],
    value=960
)

# ---------------- MODEL LOADER ----------------
@st.cache_resource(show_spinner=False)
def load_yolov5(weights_path, conf, iou):
    repo_dir = os.path.join(os.getcwd(), "yolov5")
    if not os.path.isdir(repo_dir):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5.git", repo_dir], check=True)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # Allowlist YOLOv5 DetectionModel for PyTorch 2.6+
    from models.yolo import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])

    # Load custom weights using local hub (avoids internet)
    model = torch.hub.load(repo_dir, "custom", path=weights_path, source="local", verbose=False)
    model.conf = float(conf)
    model.iou  = float(iou)
    return model

model = load_yolov5(MODEL_PATH, conf_thresh, iou_thresh)
CLASS_NAMES = model.names

# ---------------- LAYOUT ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("🖼️ Image")
    img_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="img_up")
    img_btn  = st.button("Detect People in Image")

with c2:
    st.subheader("🎥 Video")
    vid_file = st.file_uploader("Upload video", type=["mp4","mov","avi","m4v","mpeg","mpg","webm"], key="vid_up")
    vid_btn  = st.button("Detect People in Video")

with c3:
    st.subheader("🚶 Line Crossing")
    line_vid_file = st.file_uploader("Upload video", type=["mp4","mov","avi","m4v","mpeg","mpg","webm"], key="line_up")
    line_btn      = st.button("Count People Crossing Line")

center = st.container()
with center:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    frame_placeholder = st.empty()
    text_placeholder  = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def show_big_text(msg, color="black"):
    text_placeholder.markdown(
        f"<h3 style='text-align:center; color:{color};'><b>{msg}</b></h3>",
        unsafe_allow_html=True
    )

def side_of_line(p, a, b):
    return np.sign((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]))

def infer_v5(frame):
    """Run YOLOv5 on a BGR frame -> annotated RGB, dets for DeepSORT, count."""
    model.conf = float(conf_thresh)
    model.iou  = float(iou_thresh)
    # NEW: pass imgsz to reduce internal resampling blur
    results = model(frame, size=int(imgsz))
    annotated_bgr = results.render()[0]
    annotated_rgb = annotated_bgr[:, :, ::-1]

    df = results.pandas().xyxy[0]
    dets, count = [], 0
    for _, row in df.iterrows():
        if row["name"] != "person":
            continue
        x1, y1 = int(row["xmin"]), int(row["ymin"])
        x2, y2 = int(row["xmax"]), int(row["ymax"])
        conf   = float(row["confidence"])
        dets.append([[x1, y1, x2 - x1, y2 - y1], conf, "person"])
        count += 1
    return annotated_rgb, dets, count

# ---------------- OPTION 1: IMAGE ----------------
if img_file and img_btn:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(img_file.read())
        img_path = tfile.name
    img = cv2.imread(img_path)
    annotated, dets, num_people = infer_v5(img)
    frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
    show_big_text(f"Detected {num_people} people")
    os.unlink(img_path)

# ---------------- OPTION 2: VIDEO ----------------
if vid_file and vid_btn:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(vid_file.read())
        vid_path = tfile.name
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        st.error("Could not open this video. Convert to MP4 if needed.")
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, dets, frame_count = infer_v5(frame)
            frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
            show_big_text(f"Current frame: {frame_count} people")
        cap.release()
    os.unlink(vid_path)

# ---------------- OPTION 3: LINE-CROSS ----------------
if line_vid_file and line_btn:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(line_vid_file.read())
        line_vid_path = tfile.name
    cap = cv2.VideoCapture(line_vid_path)
    if not cap.isOpened():
        st.error("Could not open this video. Convert to MP4 if needed.")
    else:
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        p1, p2 = (int(0.1 * W), H // 2), (int(0.9 * W), H // 2)

        tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=0.7)
        last_side = {}
        entered, exited = 0, 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated_rgb, dets, _ = infer_v5(frame)
            tracks = tracker.update_tracks(dets, frame=frame)
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
            for t in tracks:
                if not t.is_confirmed(): continue
                tid = t.track_id
                x1, y1, x2, y2 = map(int, t.to_ltrb())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                s = side_of_line((cx, cy), p1, p2)
                if tid not in last_side:
                    last_side[tid] = s
                else:
                    prev = last_side[tid]
                    if s != 0 and prev != 0 and s != prev:
                        if prev < s: entered += 1
                        else:        exited  += 1
                    last_side[tid] = s
                cv2.rectangle(frame, (x1,y1), (x2,y2), (80,200,120), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
            show_big_text(f"Entered: {entered}   |   Exited: {exited}")
        cap.release()
    os.unlink(line_vid_path)
