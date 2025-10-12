# streamlit_unified_detection_headless.py
"""
Unified Streamlit app: Head pose + Hand gestures + Blink detection
- Uses st.camera_input instead of cv2.VideoCapture
- Mediapipe for hand/face
- Pygame audio playback
- Twilio SMS alerts
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
from math import hypot
from queue import Queue
from collections import deque

# ----------------------------
# Streamlit UI setup
# ----------------------------
st.set_page_config(page_title="Unified Gesture/Head/Blink Detection", layout="wide")
st.title("🧠 Unified: Hand + Head + Blink Detection (Streamlit)")

st.markdown("""
This app runs hand gestures, head-pose instructions, and blink detection together
in a single video feed. Use sidebar to tune features and enable voice/SMS alerts.
""")

# Sidebar controls
st.sidebar.header("Modules")
enable_hand = st.sidebar.checkbox("Enable Hand Detection", value=True)
enable_eye = st.sidebar.checkbox("Enable Blink Detection", value=True)
enable_head = st.sidebar.checkbox("Enable Head Pose Detection", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Audio / Alerts")
use_voice = st.sidebar.checkbox("Enable Voice Playback (pygame)", value=True)
use_twilio = st.sidebar.checkbox("Enable SMS Alerts (Twilio)", value=False)
tw_sid = st.sidebar.text_input("Twilio SID", value="")
tw_token = st.sidebar.text_input("Twilio Auth Token", value="", type="password")
tw_from = st.sidebar.text_input("From (Twilio number)", value="")
tw_to = st.sidebar.text_input("Caregiver number (+countrycode)", value="")

st.sidebar.markdown("---")
st.sidebar.header("Performance Tuning")
process_width = st.sidebar.slider("Process width (px)", min_value=320, max_value=1280, value=640, step=64)
model_complexity = st.sidebar.selectbox("Hand model complexity", [0, 1, 2], index=0)
refine_landmarks = st.sidebar.checkbox("Refine face landmarks (slower)", value=False)

# ----------------------------
# Audio setup
# ----------------------------
pygame_inited = False
if use_voice:
    try:
        pygame.mixer.init()
        pygame_inited = True
    except Exception as e:
        st.warning(f"Pygame init failed: {e}")

audio_queue = Queue()
audio_lock = pygame.Lock() if pygame_inited else None

def audio_worker():
    while True:
        path = audio_queue.get()
        if path is None:
            audio_queue.task_done()
            break
        if path and os.path.exists(path):
            try:
                with audio_lock:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")
        audio_queue.task_done()

if use_voice and pygame_inited:
    threading.Thread(target=audio_worker, daemon=True).start()

def enqueue_audio(path):
    if use_voice and pygame_inited:
        audio_queue.put(path)

# ----------------------------
# Mediapipe setup
# ----------------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=model_complexity,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=refine_landmarks,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ----------------------------
# Helpers (EAR, head pose, hand)
# ----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FINGER_TIPS = [8, 12, 16, 20]

def eye_aspect_ratio(landmarks, left_indices, right_indices):
    def _ear(points):
        v1 = hypot(points[1][0] - points[5][0], points[1][1] - points[5][1])
        v2 = hypot(points[2][0] - points[4][0], points[2][1] - points[4][1])
        h = hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])
        return (v1 + v2) / (2.0 * h) if h != 0 else 0.0
    left = [(landmarks[i].x, landmarks[i].y) for i in left_indices]
    right = [(landmarks[i].x, landmarks[i].y) for i in right_indices]
    return (_ear(left) + _ear(right)) / 2.0

def fingers_up(hand_landmarks):
    fingers = []
    try:
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    except:
        fingers.append(0)
    for tip in FINGER_TIPS:
        try:
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)
        except:
            fingers.append(0)
    return fingers

def detect_hand_gesture(fingers):
    if fingers == [0, 1, 0, 0, 0]: return "Bring Water"
    if fingers == [0, 1, 1, 0, 0]: return "Emergency"
    if fingers == [0, 0, 0, 0, 0]: return "Stop"
    if fingers == [0, 1, 1, 1, 0]: return "Assist me outside"
    if fingers == [0, 1, 1, 1, 1]: return "Call 108"
    if fingers == [0, 0, 1, 1, 1]: return "Contact my caregiver"
    if fingers == [0, 0, 0, 0, 1]: return "Check my supplies"
    if fingers == [0, 0, 0, 1, 1]: return "Help me sit"
    return None

# ----------------------------
# Streamlit camera input
# ----------------------------
FRAME_WINDOW = st.image([])

st.info("Use the camera input below")
camera_file = st.camera_input("Start your webcam")

if camera_file:
    file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize for processing
    h0, w0 = frame.shape[:2]
    scale = process_width / float(w0)
    frame = cv2.resize(frame, (process_width, int(h0 * scale)))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    if enable_hand:
        hand_result = hands.process(rgb)
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand_landmarks)
                gesture = detect_hand_gesture(fingers)
                if gesture:
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    st.write(f"Detected Gesture: {gesture}")
                    # enqueue_audio(GESTURE_SOUNDS.get(gesture))  # Add audio paths here if needed

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
