import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# ০. ব্রাউজার ট্যাবের টাইটেল ও আপনার ইউনিক আইকন সেট করা
# আমরা 'page_icon' হিসেবে আপনার লোগো ফাইলটিকেই ব্যবহার করছি!
st.set_page_config(
    page_title="AI Size Measurement System",
    page_icon="🎯", # আমরা ব্রাউজার ট্যাবে ইমোজি রাখলাম যেন ক্র্যাশ না করে
    layout="centered"
)
# ১. লোগো ও ব্র্যান্ডিং সেকশন
col1, col2 = st.columns([1, 4]) # লোগো এবং টাইটেল পাশাপাশি দেখানোর জন্য কলাম তৈরি

with col1:
    # গিটহাবে logo.png ফাইলটি থাকলে সেটি দেখাবে
    if os.path.exists("logo.png"):
        st.image("logo.png", width=90)
    else:
        # যদি কোনো কারণে লোগো ফাইল না পাওয়া যায়, তবে ব্যাকআপ হিসেবে একটি ইমোজি
        st.write("🎯") 

with col2:
    st.title("AI Real-Time Size Measurement System")

st.write("YOLOv8 এবং OpenCV ব্যবহার করে তৈরি একটি অবজেক্ট ডিটেকশন ও সাইজ মেজারমেন্ট অ্যাপ।")
st.markdown("---")

# ২. মডেল লোড করা
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ৩. ভিডিও আপলোড অপশন
uploaded_file = st.file_uploader("একটি ভিডিও ফাইল আপলোড করুন (MP4)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # আপলোড করা ভিডিও ফাইলটিকে সাময়িকভাবে সেভ করার জন্য একটি নির্দিষ্ট নাম দেওয়া
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
    
    st.write("🔄 এআই মডেল আপনার ভিডিও প্রসেস করছে... নিচে লাইভ আউটপুট দেখুন:")
    
    # রেজাল্ট দেখানোর জন্য একটি খালি জায়গা রাখা
    output_container = st.empty()
    
    PPCM = 12 # Pixels Per Centimeter (পরিমাপের জন্য এই মানটি ক্যালিব্রেট করা দরকার)
    
    # YOLO মডেলকে সরাসরি এই টেম্পোরারি ফাইলের পাথ (path) দেওয়া হলো
    # stream=True ব্যবহার করায় এটি ফ্রেম বাই ফ্রেম রেজাল্ট দেবে
    results = model.predict(source=temp_path, stream=True)
    
    summary_results = []
    
    # ভিডিওর ফ্রেম বাই ফ্রেম রেজাল্ট প্রসেস করা
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
                
                # সেমিতে কনভার্ট ( width_cm = width_pixel / PPCM )
                width_cm = round(width_pixel / PPCM, 2)
                height_cm = round(height_pixel / PPCM, 2)
                
                info = f"📦 **অবজেক্ট:** {object_name.upper()} | **বাস্তব সাইজ:** {width_cm} cm (প্রস্থ) x {height_cm} cm (উচ্চতা)"
                if info not in summary_results:
                    summary_results.append(info)
            
            # আমরা এখানে আর 'break' দিচ্ছি না, ফলে পুরো ভিডিওর সব অবজেক্ট ডিটেক্ট হবে
            # তবে আউটপুট কন্টেইনার আপডেট করে সব রেজাল্ট দেখাবো
            with output_container.container():
                st.write("### 📊 পরিমাপের ফলাফল (সব অবজেক্ট):")
                for res in summary_results:
                    st.markdown(res)
            
    # স্ক্রিনে ফাইনাল আউটপুট দেখানো
    st.success("🎉 প্রসেসিং সম্পন্ন হয়েছে!")
    st.markdown("---")
    st.write("### 📜 চূড়ান্ত পরিমাপের সামারি:")
    
    if len(summary_results) > 0:
        for res in summary_results:
            st.markdown(res)
    else:
        st.warning("⚠️ ভিডিওতে কোনো নির্দিষ্ট অবজেক্ট খুঁজে পাওয়া যায়নি। অনুগ্রহ করে অন্য ভিডিও ট্রাই করুন।")
        
    # কাজ শেষে টেম্পোরারি ফাইলটি সার্ভার থেকে ডিলিট করা (Memory সাশ্রয়ের জন্য)
    try:
        os.unlink(temp_path)
    except Exception as e:
        passs e:
        pass
