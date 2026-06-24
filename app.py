import streamlit as str
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile

# অ্যাপের টাইটেল ও ইন্টারফেস
st.title("🎯 AI Real-Time Size Measurement System")
st.write("YOLOv8 এবং OpenCV ব্যবহার করে তৈরি একটি অবজেক্ট ডিটেকশন ও সাইজ মেজারমেন্ট অ্যাপ।")

# ১. মডেল লোড করা
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ২. ভিডিও আপলোড অপশন
uploaded_file = st.file_uploader("একটি ভিডিও ফাইল আপলোড করুন (MP4)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # আপলোড করা ভিডিও সাময়িকভাবে সেভ করা
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    # OpenCV দিয়ে ভিডিও রিড করা
    cap = cv2.VideoCapture(tfile.name)
    
    st.write("🔄 এআই মডেল আপনার ভিডিও প্রসেস করছে... নিচে লাইভ আউটপুট দেখুন:")
    
    # রেজাল্ট দেখানোর জন্য একটি খালি জায়গা রাখা
    output_text = st.empty()
    
    PPCM = 12 # Pixels Per Centimeter
    
    # ভিডিওর ফ্রেম বাই ফ্রেম প্রসেস করা
    results = model.predict(source=tfile.name, stream=True)
    
    summary_results = []
    
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                object_name = model.names[class_id]
                
                # বক্স কোঅর্ডিনেট ও সাইজ বের করা
                x1, y1, x2, y2 = box.xyxy[0]
                width_pixel = int(x2 - x1)
                height_pixel = int(y2 - y1)
                
                # সেমিতে কনভার্ট
                width_cm = round(width_pixel / PPCM, 2)
                height_cm = round(height_pixel / PPCM, 2)
                
                info = f"📦 **অবজেক্ট:** {object_name.upper()} | **বাস্তব সাইজ:** {width_cm} cm (প্রস্থ) x {height_cm} cm (উচ্চতা)\n"
                summary_results.append(info)
            
            # আমরা এখানে জাস্ট কয়েকটা ফ্রেমের সামারি রেন্ডার করে দেখাবো
            break
            
    # স্ক্রিনে ফাইনাল আউটপুট দেখানো
    st.success("🎉 প্রসেসিং সম্পন্ন হয়েছে!")
    st.write("### 📊 পরিমাপের ফলাফল:")
    for res in summary_results:
        st.markdown(res)
