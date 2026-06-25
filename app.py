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

# ১. লোগো ও ব্র্যান্ডিং সেকশন (অনলাইন ইউআরএল)
col1, col2 = st.columns([1, 4])
with col1:
    logo_url = "https://images2.imgbox.com/39/3a/0b4ObyWd_o.png"
    st.image(logo_url, width=90)

with col2:
    st.title("AI Real-Time Size Measurement System")

st.write("YOLOv8 এবং OpenCV ব্যবহার করে তৈরি একটি রঙিন অবজেক্ট ডিটেকশন ও সাইজ মেজারমেন্ট অ্যাপ।")
st.markdown("---")

# ২. মডেল লোড করা
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ৩. ভিন্ন ভিন্ন অবজেক্টের জন্য র্যান্ডম রঙিন কালার প্যালেট তৈরি (OpenCV BGR Format)
@st.cache_data
def get_colors():
    # ৮০টি কমন অবজেক্ট ক্লাসের জন্য আলাদা আলাদা উজ্জ্বল রঙের তালিকা
    return [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

colors = get_colors()

# ৪. ভিডিও আপলোড অপশন
uploaded_file = st.file_uploader("একটি ভিডিও ফাইল আপলোড করুন (MP4)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
    
    st.write("🔄 এআই মডেল আপনার ভিডিও রঙিন ফ্রেমে প্রসেস করছে...")
    
    # ভিডিও প্লেয়ার এবং রেজাল্ট টেক্সটের জন্য দুটি আলাদা জায়গা রাখা
    video_placeholder = st.empty()
    summary_placeholder = st.empty()
    
    # OpenCV দিয়ে ভিডিও রিড করা ভিডিওর আসল ফ্রেম রেট ও সাইজ ঠিক রাখার জন্য
    cap = cv2.VideoCapture(temp_path)
    PPCM = 12 # Pixels Per Centimeter
    
    summary_results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # প্রতি ফ্রেমে YOLOv8 অবজেক্ট ডিটেকশন রান করা
        results = model(frame, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                object_name = model.names[class_id]
                color = colors[class_id] # এই অবজেক্টের জন্য নির্দিষ্ট রঙ সিলেক্ট করা
                
                # বক্স কোঅর্ডিনেট
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # সাইজ ক্যালকুলেশন
                width_pixel = x2 - x1
                height_pixel = y2 - y1
                width_cm = round(width_pixel / PPCM, 1)
                height_cm = round(height_pixel / PPCM, 1)
                
                # ১. OpenCV দিয়ে রঙিন বাউন্ডিং বক্স আঁকা (Thickness = 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # ২. রঙিন লেবেল টেক্সট তৈরি করা
                label = f"{object_name.upper()}: {width_cm}x{height_cm} cm"
                
                # টেক্সটের ব্যাকগ্রাউন্ডে একটি ছোট রঙিন বক্স দেওয়া (যাতে লেখাটি স্পষ্ট বোঝা যায়)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                
                # সাদা রঙে মেইন টেক্সটটি লেখা
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # টেক্সট সামারির জন্য ডাটা সংরক্ষণ করা
                info = f"📦 **{object_name.upper()}** | 📏 {width_cm} cm x {height_cm} cm"
                if info not in summary_results:
                    summary_results.append(info)
        
        # OpenCV-র BGR ইমেজকে স্ট্রিমলিটের জন্য RGB-তে কনভার্ট করা
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # রিয়েল-টাইমে রঙিন ভিডিও ফ্রেমটি অ্যাপ স্ক্রিনে দেখানো
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # পাশাপাশি টেক্সট রেজাল্ট আপডেট করা
        with summary_placeholder.container():
            st.write("### 📊 লাইভ পরিমাপের তালিকা:")
            for res in summary_results[-5:]: # শুধু শেষ ৫টি ডিটেকশন দেখাবে স্ক্রিন পরিষ্কার রাখতে
                st.markdown(res)
                
    cap.release()
    st.success("🎉 পুরো ভিডিওর কালারফুল প্রসেসিং সম্পন্ন হয়েছে!")
    
    # ফাইনাল ফুল সামারি
    st.markdown("---")
    st.write("### 📜 চূড়ান্ত পরিমাপের সামারি:")
    for res in summary_results:
        st.markdown(res)
        
    try:
        os.unlink(temp_path)
    except:
        pass  

