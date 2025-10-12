# streamlit_unified_headless.py
"""
Headless Streamlit App:
- Hand gestures, head pose, blink detection
- st.camera_input instead of cv2.VideoCapture
- Pygame audio playback
- Twilio SMS alerts
"""

import streamlit as st
from PIL import Image
import numpy as np

import mediapipe as mp
import pygame
import threading
import time
import os
from math import hypot
from queue import Queue
from collections import deque

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Unified Gesture/Head/Blink Detection", layout="wide")
st.title("🧠 Unified: Hand + Head + Blink Detection (Headless)")

# Sidebar toggles
st.sidebar.header("Modules")
enable_hand = st.sidebar.checkbox("Enable Hand Detection", True)
enable_eye = st.sidebar.checkbox("Enable Blink Detection", True)
enable_head = st.sidebar.checkbox("Enable Head Pose Detection", True)

st.sidebar.header("Audio / Alerts")
use_voice = st.sidebar.checkbox("Enable Voice Playback (pygame)", True)
use_twilio = st.sidebar.checkbox("Enable SMS Alerts (Twilio)", False)
tw_sid = st.sidebar.text_input("Twilio SID")
tw_token = st.sidebar.text_input("Twilio Auth Token", type="password")
tw_from = st.sidebar.text_input("From (Twilio number)")
tw_to = st.sidebar.text_input("Caregiver number (+countrycode)")

st.sidebar.header("Performance")
process_width = st.sidebar.slider("Process width (px)", 320, 1280, 640, 64)
model_complexity = st.sidebar.selectbox("Hand model complexity", [0,1,2], 0)
refine_landmarks = st.sidebar.checkbox("Refine face landmarks", False)

# ----------------------------
# Audio setup
# ----------------------------
pygame_inited = False
if use_voice:
    try:
        pygame.mixer.init()
        pygame_inited = True
    except:
        st.warning("Pygame mixer init failed")

audio_queue = Queue()
audio_lock = threading.Lock() if pygame_inited else None

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
# Helper constants
# ----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FINGER_TIPS = [8, 12, 16, 20]

EAR_THRESHOLD = 0.3
CONSEC_FRAMES = 4
LONG_BLINK_SECONDS = 2.0
DOUBLE_BLINK_MAX_INTERVAL = 0.8

YAW_RIGHT_THRESH = 10
YAW_LEFT_THRESH = -10
PITCH_UP_THRESH = -12
PITCH_DOWN_THRESH = 12
STABLE_TIME = 0.6

# ----------------------------
# Helpers
# ----------------------------
def eye_aspect_ratio(landmarks, left_indices, right_indices):
    def _ear(points):
        v1 = hypot(points[1][0]-points[5][0], points[1][1]-points[5][1])
        v2 = hypot(points[2][0]-points[4][0], points[2][1]-points[4][1])
        h = hypot(points[0][0]-points[3][0], points[0][1]-points[3][1])
        return (v1+v2)/(2.0*h) if h != 0 else 0
    left = [(landmarks[i].x, landmarks[i].y) for i in left_indices]
    right = [(landmarks[i].x, landmarks[i].y) for i in right_indices]
    return (_ear(left)+_ear(right))/2.0

def fingers_up(hand_landmarks):
    fingers = [1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0]
    for tip in FINGER_TIPS:
        try:
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y else 0)
        except:
            fingers.append(0)
    return fingers

def detect_hand_gesture(fingers):
    gestures = {
        (0,1,0,0,0): "Bring Water",
        (0,1,1,0,0): "Emergency",
        (0,0,0,0,0): "Stop",
        (0,1,1,1,0): "Assist me outside",
        (0,1,1,1,1): "Call 108",
        (0,0,1,1,1): "Contact my caregiver",
        (0,0,0,0,1): "Check my supplies",
        (0,0,0,1,1): "Help me sit",
    }
    return gestures.get(tuple(fingers))

# ----------------------------
# Streamlit camera input
# ----------------------------
FRAME_WINDOW = st.image([])

camera_file = st.camera_input("Start your webcam")
if camera_file:
    img = Image.open(camera_file)
    frame = np.array(img)  # RGB format
    # Resize
    h0, w0 = frame.shape[:2]
    scale = process_width / float(w0)
    new_h = int(h0*scale)
    frame = np.array(img.resize((process_width, new_h)))
    rgb = frame  # Already RGB

    # Hand detection
    if enable_hand:
        hand_result = hands.process(rgb)
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand_landmarks)
                gesture = detect_hand_gesture(fingers)
                if gesture:
                    st.write(f"Detected Gesture: {gesture}")
                    # enqueue_audio(GESTURE_SOUNDS.get(gesture))

    # Placeholder display
    FRAME_WINDOW.image(frame)
