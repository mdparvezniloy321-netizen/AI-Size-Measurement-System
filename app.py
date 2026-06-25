import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import random

# ০. Page Config setup
st.set_page_config(
    page_title="AI Size Measurement System",
    page_icon="🎯",
    layout="centered"
)

# ১. লোগো ও কালারফুল ব্র্যান্ডিং সেকশন (HTML + CSS হাইলাইট)
col1, col2 = st.columns([1, 4])
with col1:
    logo_url = "https://images2.imgbox.com/39/3a/0b4ObyWd_o.png"
    st.image(logo_url, width=90)

with col2:
    # টাইটেলকে গ্রাডিয়েন্ট কালারফুল করা হয়েছে
    st.markdown("""
        <h1 style='text-align: left; color: #FF4B4B; font-family: sans-serif; font-size: 32px; margin-bottom: 0px;'>
            <span style='background: linear-gradient(to right, #FF416C, #FF4B2B); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                AI Real-Time Size Measurement System
            </span>
        </h1>
    """, unsafe_allow_html=True)

# সাবটাইটেলের নির্দিষ্ট কি-ওয়ার্ডগুলো (YOLOv8, OpenCV, অবজেক্ট ডিটেকশন, সাইজ মেজারমেন্ট) কালারফুল হাইলাইট
st.markdown("""
    <p style='font-size: 16px; color: #555555; line-height: 1.6;'>
        <b style='color: #00A86B;'>YOLOv8</b> এবং <b style='color: #007ACC;'>OpenCV</b> ব্যবহার করে তৈরি একটি 
        <span style='background-color: #FFEB3B; padding: 2px 6px; border-radius: 4px; color: #000000; font-weight: bold;'>রঙিন অবজেক্ট ডিটেকশন</span> 
        ও <span style='background-color: #E1F5FE; padding: 2px 6px; border-radius: 4px; color: #0288D1; font-weight: bold;'>সাইজ মেজারমেন্ট</span> অ্যাপ।
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ২. মডেল লোড করা
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ৩. ভিন্ন ভিন্ন অবজেক্টের জন্য র্যান্ডম রঙিন কালার প্যালেট তৈরি
@st.cache_data
def get_colors():
    return [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

colors = get_colors()

# ৪. ভিডিও আপলোড অপশন
uploaded_file = st.file_uploader("একটি ভিডিও ফাইল আপলোড করুন (MP4)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
    
    st.write("🔄 এআই মডেল আপনার ভিডিও রঙিন ফ্রেমে প্রসেস করছে...")
    
    video_placeholder = st.empty()
    summary_placeholder = st.empty()
    
    cap = cv2.VideoCapture(temp_path)
    PPCM = 12 # Pixels Per Centimeter
    
    summary_results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                object_name = model.names[class_id]
                color = colors[class_id]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                width_pixel = x2 - x1
                height_pixel = y2 - y1
                width_cm = round(width_pixel / PPCM, 1)
                height_cm = round(height_pixel / PPCM, 1)
                
                # রঙিন বক্স এবং টেক্সট আঁকা
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{object_name.upper()}: {width_cm}x{height_cm} cm"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                info = f"📦 **{object_name.upper()}** | 📏 {width_cm} cm x {height_cm} cm"
                if info not in summary_results:
                    summary_results.append(info)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        with summary_placeholder.container():
            st.write("### 📊 লাইভ পরিমাপের তালিকা:")
            for res in summary_results[-5:]:
                st.markdown(res)
                
    cap.release()
    st.success("🎉 পুরো ভিডিওর কালারফুল প্রসেসিং সম্পন্ন হয়েছে!")
    
    st.markdown("---")
    st.write("### 📜 চূড়ান্ত পরিমাপের সামারি:")
    for res in summary_results:
        st.markdown(res)
        
    try:
        os.unlink(temp_path)
    except:
        pass

