import cv2
import torch
import time
import pygame
import os
import numpy as np
from deepface import DeepFace
from tkinter import Tk, Button, filedialog

# Initialize alarm
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model.eval()
model.conf = 0.5

# Globals
person_faces = {}
person_colors = {}
bag_owners = {}
bag_left_times = {}
alarm_played_for = set()

ALERT_TIME = 10  # seconds
PROCESS_EVERY_N_FRAMES = 3  # Process every Nth frame

# Color fallback
def get_dominant_color(image):
    image = cv2.resize(image, (50, 50))
    data = image.reshape((-1, 3))
    data = np.float32(data)
    _, labels, palette = cv2.kmeans(data, 1, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS)
    return tuple(palette[0].astype(int))

# Play alarm and save snapshot
def trigger_alert(bag_id, frame, frame_count):
    if bag_id in alarm_played_for:
        return
    pygame.mixer.music.play()
    filename = f"snapshot_{bag_id}_{frame_count}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[ALERT] Bag {bag_id} stolen! Snapshot saved as {filename}")
    alarm_played_for.add(bag_id)

def color_similarity(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2)) < 60

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter("output_annotated.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w, h))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue  # Skip frame to boost performance

        # Resize for speed (optional)
        # small_frame = cv2.resize(frame, (640, 360))

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        persons, bags = [], []

        for det in detections:
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            cls_name = model.names[int(cls)]
            if cls_name == 'person':
                persons.append((x1, y1, x2, y2))
            elif cls_name in ['backpack', 'handbag', 'suitcase']:
                bags.append((x1, y1, x2, y2))

        current_people = {}
        for i, (x1, y1, x2, y2) in enumerate(persons):
            pid = f"P_{x1}_{y1}"
            crop = frame[y1:y2, x1:x2]
            if pid not in person_faces:
                try:
                    rep = DeepFace.represent(crop, enforce_detection=False)
                    person_faces[pid] = rep[0]['embedding']
                except:
                    person_faces[pid] = None
                person_colors[pid] = get_dominant_color(crop)
            current_people[pid] = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, pid, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        for i, (bx1, by1, bx2, by2) in enumerate(bags):
            bag_id = f"B_{bx1}_{by1}"
            owner_found = False
            for pid, (x1, y1, x2, y2) in current_people.items():
                if abs(bx1 - x1) < 100 and abs(by1 - y2) < 100:
                    bag_owners[bag_id] = pid
                    bag_left_times[bag_id] = None
                    owner_found = True
                    break

            if not owner_found:
                if bag_id not in bag_left_times:
                    bag_left_times[bag_id] = time.time()
                elif time.time() - bag_left_times[bag_id] > ALERT_TIME:
                    owner = bag_owners.get(bag_id)
                    for pid in current_people:
                        if pid == owner:
                            continue
                        same_face = False
                        same_color = False
                        if person_faces.get(pid) and person_faces.get(owner):
                            dist = np.linalg.norm(
                                np.array(person_faces[pid]) -
                                np.array(person_faces[owner]))
                            same_face = dist < 0.65
                        if color_similarity(person_colors[pid],
                                            person_colors.get(owner, (0,0,0))):
                            same_color = True
                        if not (same_face or same_color):
                            trigger_alert(bag_id, frame, frame_count)

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            label = bag_owners.get(bag_id, "Unowned")
            cv2.putText(frame, f"{bag_id} ({label})", (bx1, by1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        out.write(frame)
        cv2.imshow("Alert Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# GUI to upload video
root = Tk()
root.title("Unattended Object Alert System")
root.geometry("300x150")

def upload_video():
    path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if path:
        print(f"Processing: {path}")
        process_video(path)

Button(root, text="Upload Video", command=upload_video,
       font=("Arial", 14), bg="lightgreen").pack(expand=True)

root.mainloop()
