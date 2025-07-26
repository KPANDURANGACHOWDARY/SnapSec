import cv2
import torch
import time
import pygame
from deepface import DeepFace
import numpy as np
import uuid
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model.conf = 0.5

# Initialize webcam (reduced resolution for smoother processing)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Alarm setup
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alert.wav")
alarm_sound.set_volume(1.0)

# Create folders to save snapshots
os.makedirs("snapshots/unattended", exist_ok=True)
os.makedirs("snapshots/theft", exist_ok=True)

# Constants
ALERT_TRIGGER_TIME = 10  # seconds
IOU_THRESHOLD = 0.6
STABILITY_THRESHOLD = 30
TARGET_CLASSES = ['backpack', 'handbag', 'suitcase']
FACE_CLASS = 'person'

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# State tracking
object_appear_time = {}
object_locations = {}
ownership_mapping = {}  # object_id -> person_id
alerted_bags = set()
person_profiles = {}  # person_id -> {face_encoding, shirt_color}

# Helpers
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    xi1 = max(x1, a1)
    yi1 = max(y1, b1)
    xi2 = min(x2, a2)
    yi2 = min(y2, b2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (a2 - a1) * (b2 - b1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def is_stable(box1, box2):
    return all(abs(a - b) < STABILITY_THRESHOLD for a, b in zip(box1, box2))

def get_face_encoding(face_img):
    try:
        result = DeepFace.represent(face_img, enforce_detection=False)
        if result and isinstance(result, list):
            return result[0]["embedding"]
    except:
        return None
    return None

def compare_faces(enc1, enc2, threshold=7.0):
    if enc1 is None or enc2 is None:
        return float('inf')
    return np.linalg.norm(np.array(enc1) - np.array(enc2))

def get_dominant_color(img):
    if img is None or img.size == 0:
        return (0, 0, 0)
    avg_color = cv2.mean(img)[:3]
    return tuple(map(int, avg_color))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]
    current_objects = []
    persons = []

    for *xyxy, conf, cls in detections:
        class_name = results.names[int(cls)]
        x1, y1, x2, y2 = map(int, xyxy)
        box = [x1, y1, x2, y2]

        if class_name == FACE_CLASS:
            persons.append(box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if class_name not in TARGET_CLASSES:
            continue

        matched_id = None
        for obj_id, prev_box in object_locations.items():
            if iou(box, prev_box) > IOU_THRESHOLD and is_stable(box, prev_box):
                matched_id = obj_id
                break

        if matched_id is None:
            matched_id = str(uuid.uuid4())
            object_appear_time[matched_id] = time.time()

        object_locations[matched_id] = box
        current_objects.append(matched_id)

        duration = time.time() - object_appear_time[matched_id]
        label = f"Bag ({int(duration)}s)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        owner_found = False
        for person_box in persons:
            px1, py1, px2, py2 = person_box
            person_crop = frame[py1:py2, px1:px2]

            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            face_enc = None
            if len(faces) > 0:
                (fx, fy, fw, fh) = faces[0]
                face_crop = person_crop[fy:fy+fh, fx:fx+fw]
                face_enc = get_face_encoding(face_crop)

            shirt_crop = person_crop[int((py2 - py1) * 0.4):, :]
            shirt_color = get_dominant_color(shirt_crop)

            matched_person_id = None
            for pid, profile in person_profiles.items():
                if face_enc and compare_faces(face_enc, profile['face']) < 7:
                    matched_person_id = pid
                    break
                elif not face_enc and np.linalg.norm(np.array(shirt_color) - np.array(profile['shirt'])) < 50:
                    matched_person_id = pid
                    break

            if not matched_person_id:
                matched_person_id = str(uuid.uuid4())
                person_profiles[matched_person_id] = {'face': face_enc, 'shirt': shirt_color}

            ownership_mapping[matched_id] = matched_person_id
            owner_found = True
            break

        if not owner_found and duration > ALERT_TRIGGER_TIME and matched_id not in alerted_bags:
            print("\n⚠️ ALERT: Unattended object detected.")
            alarm_sound.play()
            alerted_bags.add(matched_id)
            snapshot_path = f"snapshots/unattended/{matched_id}.jpg"
            cv2.imwrite(snapshot_path, frame)

    # Theft detection
    for person_box in persons:
        px1, py1, px2, py2 = person_box
        person_crop = frame[py1:py2, px1:px2]
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        face_enc = None
        if len(faces) > 0:
            (fx, fy, fw, fh) = faces[0]
            face_crop = person_crop[fy:fy+fh, fx:fx+fw]
            face_enc = get_face_encoding(face_crop)

        shirt_crop = person_crop[int((py2 - py1) * 0.4):, :]
        shirt_color = get_dominant_color(shirt_crop)

        for bag_id, bag_box in object_locations.items():
            bx1, by1, bx2, by2 = bag_box
            overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
            overlap_y = max(0, min(py2, by2) - max(py1, by1))
            if overlap_x * overlap_y > 0:
                owner_id = ownership_mapping.get(bag_id)
                if owner_id:
                    profile = person_profiles[owner_id]
                    face_sim = compare_faces(face_enc, profile['face']) if face_enc else float('inf')
                    shirt_sim = np.linalg.norm(np.array(shirt_color) - np.array(profile['shirt']))
                    if (face_sim > 8 or shirt_sim > 60) and bag_id not in alerted_bags:
                        print("\n🚨 THEFT ALERT: Another person is taking the bag!")
                        alarm_sound.play()
                        alerted_bags.add(bag_id)
                        snapshot_path = f"snapshots/theft/{bag_id}_{int(time.time())}.jpg"
                        cv2.imwrite(snapshot_path, frame)

    # Cleanup
    for obj_id in list(object_locations.keys()):
        if obj_id not in current_objects:
            object_locations.pop(obj_id)
            object_appear_time.pop(obj_id, None)
            ownership_mapping.pop(obj_id, None)
            alerted_bags.discard(obj_id)

    cv2.imshow("Unattended Object Alert System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
