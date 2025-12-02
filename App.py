import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time

st.set_page_config(page_title="YOLO Real-Time Video", layout="centered")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("⚡ Real-Time YOLO Video Detection")
st.write("Upload a video and watch YOLO detect objects **live**, frame-by-frame.")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video:

    # Save uploaded file temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input.write(uploaded_video.read())
    temp_video_path = temp_input.name

    st.video(temp_video_path)

    if st.button("▶ Start Real-Time Detection"):
        st.write("🚀 Running YOLO in real-time. Please wait...")

        # Placeholder where frames will be shown
        frame_window = st.empty()

        cap = cv2.VideoCapture(temp_video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run YOLO
            results = model(frame)
            annotated_frame = results[0].plot()

            # Convert for Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Show frame in realtime
            frame_window.image(rgb_frame, channels="RGB")

            # Small delay for UI refresh
            time.sleep(0.02)

        cap.release()
        st.success("✔ Video Finished")
