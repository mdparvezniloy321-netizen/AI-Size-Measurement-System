import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# ০. Page Config setup (ট্যাবে ইমোজি থাকবে যেন ক্র্যাশ না করে)
st.set_page_config(
    page_title="AI Size Measurement System",
    page_icon="🎯",
    layout="centered"
)

# ১. লোগো ও ব্র্যান্ডিং সেকশন (অনলাইন ইউআরএল ট্রিক)
col1, col2 = st.columns([1, 4])

with col1:
    # এখানে সরাসরি ইন্টারনেটের লাইভ লিংক বসিয়ে দেওয়া হয়েছে, তাই ক্র্যাশ করবে না
    logo_url = "https://images2.imgbox.com/39/3a/0b4ObyWd_o.png"
    st.image(logo_url, width=90)

with col2:
    st.title("AI Real-Time Size Measurement System")

st.write(" Object detection and measuring by using YOLOv8 and OpenCV ")
st.markdown("---")

# ২. মডেল লোড করা
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ৩. ভিডিও আপলোড অপশন
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
