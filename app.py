import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# ০. Page Config setup
st.set_page_config(
    page_title="AI Size Measurement System",
    page_icon="🎯",
    layout="centered"
)

# ১. Branding Section (Ekhane logo.png r dorkar nei, tai crash korbe na)
st.title("🎯 AI Real-Time Size Measurement System")
st.write("YOLOv8 and OpenCV use kore toiri kora Object Detection & Size Measurement App.")
st.markdown("---")

# ২. Model Load kora
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ৩. Video Upload Option
uploaded_file = st.file_uploader("Upload a video file (MP4)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
    
    st.write("🔄 AI Model is processing your video... Check live output below:")
    
    output_container = st.empty()
    PPCM = 12 # Pixels Per Centimeter
    
    results = model.predict(source=temp_path, stream=True)
    summary_results = []
    
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                object_name = model.names[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0]
                width_pixel = int(x2 - x1)
                height_pixel = int(y2 - y1)
                
                width_cm = round(width_pixel / PPCM, 2)
                height_cm = round(height_pixel / PPCM, 2)
                
                info = f"📦 **Object:** {object_name.upper()} | **Size:** {width_cm} cm (Width) x {height_cm} cm (Height)"
                if info not in summary_results:
                    summary_results.append(info)
            
            with output_container.container():
                st.write("### 📊 Measurement Results:")
                for res in summary_results:
                    st.markdown(res)
            
    st.success("🎉 Processing Completed!")
    st.markdown("---")
    st.write("### 📜 Final Summary:")
    
    if len(summary_results) > 0:
        for res in summary_results:
            st.markdown(res)
    else:
        st.warning("⚠️ No objects detected in the video.")
        
    try:
        os.unlink(temp_path)
    except:
        pass
        pass
