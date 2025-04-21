import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import csv
import threading
from datetime import datetime
from playsound import playsound
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Configuration
EAR_THRESHOLD = 0.25
SLEEP_FRAMES_THRESHOLD = 15
DISTRACTED_POSE_THRESHOLD = 20

# Initialize timers and counters
sleep_counter = 0
focus_time = 0
distracted_time = 0
sleeping_time = 0
last_state = "ENGAGED"
state_start_time = time.time()

# Initialize MediaPipe and YOLOv10
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
model = YOLO("yolov10n.pt")

# Utility functions
def eye_aspect_ratio(landmarks):
    left = np.array([landmarks[362].x, landmarks[362].y])
    right = np.array([landmarks[263].x, landmarks[263].y])
    top = np.array([landmarks[386].x, landmarks[386].y])
    bottom = np.array([landmarks[374].x, landmarks[374].y])
    hor_dist = np.linalg.norm(left - right)
    ver_dist = np.linalg.norm(top - bottom)
    return ver_dist / hor_dist

def get_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[287].x * w, landmarks[287].y * h),
        (landmarks[57].x * w, landmarks[57].y * h),
    ], dtype="double")

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    _, rot_vec, _, _ = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rot_vec)
    pitch = np.arcsin(-rmat[2][0]) * (180 / np.pi)
    return pitch

def log_status(status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("personal_log.csv", "a", newline='') as f:
        csv.writer(f).writerow([timestamp, status])
    print(f"{timestamp} | {status}")

def play_sound(file):
    threading.Thread(target=playsound, args=(file,), daemon=True).start()

def format_time(seconds):
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes}m {seconds}s"

def create_pie_chart(focus, distract, sleep):
    import math

    labels = ['Focus', 'Distracted', 'Sleeping']
    times = [focus, distract, sleep]
    colors = ['#00cc66', '#ff9900', '#ff3300']

    print("📊 Times => Focus:", focus, "Distracted:", distract, "Sleeping:", sleep)

    # 🌐 Handle NaN or all-zero drama
    if any(math.isnan(t) or t < 0 for t in times):
        print("⚠️ Invalid data in times – Pie chart cancelled ✂️")
        return np.zeros((250, 250, 3), dtype=np.uint8)  # dummy black image

    if sum(times) == 0:
        print("🤷 All values are zero. Skipping pie chart.")
        return np.zeros((250, 250, 3), dtype=np.uint8)  # dummy black image

    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=100)
    ax.pie(times, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    buf = BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)

    pie_img = Image.open(buf)
    return cv2.cvtColor(np.array(pie_img), cv2.COLOR_RGBA2BGR)


# Main loop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_face)

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            ear = eye_aspect_ratio(landmarks)
            pitch = get_head_pose(landmarks, frame.shape)

            if ear < EAR_THRESHOLD and abs(pitch) > DISTRACTED_POSE_THRESHOLD:
                sleep_counter += 1
                if sleep_counter == SLEEP_FRAMES_THRESHOLD:
                    log_status("SLEEPING")
                    play_sound("sleepy_alert.mp3")
                status = "SLEEPING"
                color = (0, 0, 255)
            elif abs(pitch) > DISTRACTED_POSE_THRESHOLD:
                status = "DISTRACTED"
                color = (0, 165, 255)
                log_status("DISTRACTED")
                play_sound("distracted_alert.mp3")
                sleep_counter = 0
            else:
                status = "ENGAGED"
                color = (0, 255, 0)
                sleep_counter = 0

            # State duration tracking
            current_time = time.time()
            if status != last_state:
                duration = current_time - state_start_time
                if last_state == "DISTRACTED":
                    distracted_time += duration
                elif last_state == "SLEEPING":
                    sleeping_time += duration
                elif last_state == "ENGAGED":
                    focus_time += duration
                last_state = status
                state_start_time = current_time

            cv2.putText(annotated_frame, status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

    # Update pie chart
    pie = create_pie_chart(focus_time, distracted_time, sleeping_time)
    h, w = pie.shape[:2]
    frame_h, frame_w = annotated_frame.shape[:2]
    overlay_x = frame_w - w - 10
    overlay_y = 10
    annotated_frame[overlay_y:overlay_y+h, overlay_x:overlay_x+w] = pie

    # Optional text display
    cv2.putText(annotated_frame, f"Focus: {format_time(focus_time)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Distracted: {format_time(distracted_time)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(annotated_frame, f"Sleeping: {format_time(sleeping_time)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Personal Focus Monitor", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Log final state time
duration = time.time() - state_start_time
if last_state == "DISTRACTED":
    distracted_time += duration
elif last_state == "SLEEPING":
    sleeping_time += duration
elif last_state == "ENGAGED":
    focus_time += duration

cap.release()
cv2.destroyAllWindows()