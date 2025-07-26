# SnapSec
An AI-based surveillance system that detects unattended bags and thefts in real-time. It uses YOLOv5m for object/person detection and DeepFace for face recognition. The system alerts when a bag is left too long or taken by someone else, using face and clothing color to verify identity.

# 🎒 Unattended Object and Theft Detection System

An AI-powered surveillance system that detects unattended bags and possible thefts in real-time using object detection and face recognition.

## 📌 Project Description

This system monitors a video feed (webcam or CCTV) to detect if a bag is left unattended or picked up by a person other than its original owner. It uses **YOLOv5m** for detecting people and bags, and **DeepFace** for face recognition. Shirt color is used as a fallback when face is not clearly visible. The system triggers alerts and saves snapshots during suspicious activities.

## 🚀 Features

- 🔍 Real-time detection of people and bags
- ⏱️ Alerts if a bag is left unattended beyond a threshold (e.g., 10 seconds)
- 🧠 Face recognition with fallback to shirt color
- 🚨 Theft alert if a different person takes the bag
- 🖼️ Snapshot saving of alert incidents
- 🔊 Audio alarm using Pygame

## 🧠 Technologies Used

- [YOLOv5m](https://github.com/ultralytics/yolov5) – for object detection
- [DeepFace](https://github.com/serengil/deepface) – for face recognition
- OpenCV – for image processing
- Pygame – for alarm audio
- Haar cascades – for initial face detection

## 📂 Directory Structure

📁 snapshots/
┣ 📁 unattended/ # Saved images when bag is left alone
┗ 📁 theft/ # Saved images when theft is detected

🔊 alert.wav # Alarm sound
📄 main.py # Main detection script


Install Dependencies

pip install -r requirements.txt

Or manually install:

pip install torch torchvision torchaudio
pip install opencv-python pygame deepface

Run the Script

python main.py

Ensure your webcam is connected and an alert.wav sound file is present in the same directory.

⚙️ Configuration
You can change key parameters in main.py:

ALERT_TRIGGER_TIME = 10 → Adjust unattended time threshold

IOU_THRESHOLD → Controls box matching sensitivity

STABILITY_THRESHOLD → Controls detection jitter

📸 Demo Output
"Unattended Bag Detected" → Audio alert + snapshot saved

"Theft Alert" → Audio alert + snapshot of thief + object

🧪 Future Enhancements
Cloud snapshot uploads

Email/Telegram alerts

Tracking across multiple cameras

Edge-device optimization (Jetson/Raspberry Pi)

🙋‍♂️ Authors
K Pandu Ranga Chowdary

B.Tech AI/ML Project – 2025

📄 License
This project is for academic and educational use only.
