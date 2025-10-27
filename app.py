import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# -------------------------
# Streamlit Page Config & CSS
# -------------------------
st.set_page_config(
    page_title="Car & Person Detector",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .result-text {
        font-size: 1.5rem;
        font-weight: 500;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .image-container {
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(237, 242, 247, 0.5);
    }
    
    .app-footer {
        text-align: center;
        margin-top: 2rem;
        opacity: 0.7;
    }
    
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #1E40AF;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Helper Functions
# -------------------------
def is_blue_car(roi_bgr, blue_thresh=0.15):
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_fraction = (mask > 0).sum() / mask.size
    return blue_fraction >= blue_thresh

def detect_and_annotate(img_bgr, model, car_cls, person_cls, conf_thresh=0.35):
    results = model.predict(source=img_bgr, conf=conf_thresh)
    annotated = img_bgr.copy()
    car_count = 0
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == car_cls:
                car_count += 1
                roi = img_bgr[y1:y2, x1:x2]
                is_blue = is_blue_car(roi)
                color = (0, 0, 255) if is_blue else (255, 0, 0)
                label = f"Car (BLUE) {conf:.2f}" if is_blue else f"Car {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            elif cls == person_cls:
                person_count += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(annotated, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.putText(annotated, f"Cars: {car_count} | People: {person_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    return annotated, {"cars": car_count, "people": person_count}

# -------------------------
# Main App
# -------------------------
def main():
    st.markdown('<div class="main-header">Car & Person Detector</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">Upload Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Load YOLO model (cached)
    @st.cache_resource
    def load_yolo_model():
        return YOLO("models/Yolomodel.pt")  # adjust path if needed

    model = load_yolo_model()
    names = model.names
    car_cls = [k for k,v in names.items() if v=="car"][0]
    person_cls = [k for k,v in names.items() if v=="person"][0]

    if uploaded_files and st.button("Detect Cars & People"):
        with st.spinner("Processing images..."):
            for i, uploaded_file in enumerate(uploaded_files):
                with st.container():
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.markdown(f"<h3>Image {i+1}</h3>", unsafe_allow_html=True)

                    col1, col2 = st.columns([1,1])
                    image = Image.open(uploaded_file)
                    col1.image(image, caption=f"Image {i+1}: {uploaded_file.name}", use_column_width=True)

                    # Convert to OpenCV BGR format
                    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    annotated, counts = detect_and_annotate(img_bgr, model, car_cls, person_cls)

                    col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)
                    col2.markdown(f'<div class="result-text">Cars: {counts["cars"]} | People: {counts["people"]}</div>', unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
                    if i < len(uploaded_files)-1:
                        st.markdown("<hr>", unsafe_allow_html=True)

    elif st.button("Detect Cars & People"):
        st.info("Please upload one or more images first.")

    


if __name__ == "__main__":
    main()
