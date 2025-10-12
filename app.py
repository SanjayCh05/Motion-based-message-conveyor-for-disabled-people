# streamlit_unified_detection.py
"""
Unified Streamlit app: Head pose + Hand gestures + Blink detection
- Single combined video feed
- Pygame audio playback (background thread + queue)
- Streamlit sidebar toggles (head/hand/blink/voice/sms)
- Twilio SMS alerts (if enabled & credentials provided)
"""

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import threading
import os
import pygame
from math import hypot
from queue import Queue
from collections import deque

# ----------------------------
# Streamlit UI setup
# ----------------------------
st.set_page_config(page_title="Unified Gesture/Head/Blink Detection", layout="wide")
st.title("🧠 Unified: Hand + Head + Blink Detection (Streamlit)")

st.markdown(
    """
This app runs hand gestures, head-pose instructions and blink detection together
in a single video feed. Use sidebar to tune features and enable voice/SMS alerts.
"""
)

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
st.sidebar.markdown("---")
st.sidebar.caption("Press 'Start Camera' to begin")

run = st.sidebar.checkbox("🎥 Start Camera", value=False)

# ----------------------------
# Audio files
# ----------------------------
AUDIO_PATHS = {
    "Up": r"C:\Users\Chintala sanjay\Downloads\Audios\bring water.mp3",
    "Down": r"C:\Users\Chintala sanjay\Downloads\Audios\emergency1.opus",
    "Right": r"C:\Users\Chintala sanjay\Downloads\Audios\hungry.opus",
    "Left": r"C:\Users\Chintala sanjay\Downloads\Audios\i_need_firstaid.opus",
    "Center": None
}

GESTURE_SOUNDS = {
    "Bring Water": r"C:\Users\Chintala sanjay\Downloads\Audios\bring_water.opus",
    "Emergency": r"C:\Users\Chintala sanjay\Downloads\Audios\emergency.mp3",
    "Stop": r"C:\Users\Chintala sanjay\Downloads\Audios\stop.mp3",
    "Assist me outside": r"C:\Users\Chintala sanjay\Downloads\Audios\assist me outside.mp3",
    "Call 108": r"C:\Users\Chintala sanjay\Downloads\Audios\call 108.mp3",
    "Contact my caregiver": r"C:\Users\Chintala sanjay\Downloads\Audios\contact my caregiver.mp3",
    "Check my supplies": r"C:\Users\Chintala sanjay\Downloads\Audios\check my supplies.mp3",
    "Help me sit": r"C:\Users\Chintala sanjay\Downloads\Audios\check my supplies.mp3",
}

SINGLE_BLINK_AUDIO = r"C:\Users\Chintala sanjay\Downloads\Audios\call_for_caretaker.opus"
DOUBLE_BLINK_AUDIO = r"C:\Users\Chintala sanjay\Downloads\Audios\need_help.opus"

# ----------------------------
# Thresholds / Params
# ----------------------------
EAR_THRESHOLD = 0.3
CONSEC_FRAMES = 4
LONG_BLINK_SECONDS = 2.0
DOUBLE_BLINK_MAX_INTERVAL = 0.8

YAW_RIGHT_THRESH = 10
YAW_LEFT_THRESH = -10
PITCH_UP_THRESH = -12
PITCH_DOWN_THRESH = 12

STABLE_TIME = 0.6
BLINK_DEBOUNCE = 0.25
FINGER_TIPS = [8, 12, 16, 20]

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
# Pygame audio setup
# ----------------------------
pygame_inited = False
if use_voice:
    try:
        pygame.mixer.init()
        pygame_inited = True
    except Exception as e:
        st.warning(f"Pygame mixer init failed: {e}")
        pygame_inited = False

audio_queue = Queue()
audio_lock = threading.Lock()

def audio_worker():
    while True:
        path = audio_queue.get()
        if path is None:
            audio_queue.task_done()
            break
        if not path:
            audio_queue.task_done()
            continue
        try:
            with audio_lock:
                if os.path.exists(path):
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play(loops=0)
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
        audio_queue.task_done()

audio_thread = None
if use_voice and pygame_inited:
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

def enqueue_audio(path):
    if use_voice and pygame_inited:
        audio_queue.put(path if path and os.path.exists(path) else None)

def stop_audio():
    if pygame_inited:
        with audio_lock:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

# ----------------------------
# Helpers (EAR, head pose, hand gestures)
# ----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, left_indices, right_indices):
    def _ear(points):
        v1 = hypot(points[1][0] - points[5][0], points[1][1] - points[5][1])
        v2 = hypot(points[2][0] - points[4][0], points[2][1] - points[4][1])
        h = hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])
        return (v1 + v2) / (2.0 * h) if h != 0 else 0.0
    left = [(landmarks[i].x, landmarks[i].y) for i in left_indices]
    right = [(landmarks[i].x, landmarks[i].y) for i in right_indices]
    return (_ear(left) + _ear(right)) / 2.0

def get_head_pose(landmarks, img_w, img_h):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    try:
        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),
            (landmarks[152].x * img_w, landmarks[152].y * img_h),
            (landmarks[33].x * img_w, landmarks[33].y * img_h),
            (landmarks[263].x * img_w, landmarks[263].y * img_h),
            (landmarks[61].x * img_w, landmarks[61].y * img_h),
            (landmarks[291].x * img_w, landmarks[291].y * img_h)
        ], dtype="double")
    except Exception:
        return None, None, None

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except Exception:
        return None, None, None
    if not success:
        return None, None, None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = eulerAngles.flatten()
    return yaw, pitch, roll

def get_direction_from_pose(yaw, pitch):
    if yaw is None or pitch is None:
        return "Center"
    if yaw > YAW_RIGHT_THRESH:
        return "Right"
    elif yaw < YAW_LEFT_THRESH:
        return "Left"
    elif pitch > PITCH_DOWN_THRESH:
        return "Down"
    elif pitch < PITCH_UP_THRESH:
        return "Up"
    else:
        return "Center"

def fingers_up(hand_landmarks):
    fingers = []
    try:
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    except Exception:
        fingers.append(0)
    for tip in FINGER_TIPS:
        try:
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)
        except Exception:
            fingers.append(0)
    return fingers

def detect_hand_gesture(fingers):
    mapping = {
        (0,1,0,0,0): "Bring Water",
        (0,1,1,0,0): "Emergency",
        (0,0,0,0,0): "Stop",
        (0,1,1,1,0): "Assist me outside",
        (0,1,1,1,1): "Call 108",
        (0,0,1,1,1): "Contact my caregiver",
        (0,0,0,0,1): "Check my supplies",
        (0,0,0,1,1): "Help me sit"
    }
    return mapping.get(tuple(fingers), None)

# ----------------------------
# State variables
# ----------------------------
frame_times = deque(maxlen=30)
head_last_detected = "Center"
head_stable = "Center"
head_detect_time = time.time()
last_gesture = None
gesture_last_time = 0
blink_frame_counter = 0
pending_blink_time = None
total_blinks = 0
single_blink_events = 0
double_blink_events = 0
long_blink_events = 0
last_blink_event = None
last_blink_event_time = 0
yaw_q = deque(maxlen=5)
pitch_q = deque(maxlen=5)
last_command = None
last_sms_time = 0
sms_cooldown = 30

# ----------------------------
# Streamlit placeholders
# ----------------------------
FRAME_WINDOW = st.image([])
status_placeholder = st.empty()
fps_placeholder = st.empty()

# ----------------------------
# Main loop using st.camera_input()
# ----------------------------
try:
    while run:
        loop_start = time.time()
        img_file_buffer = st.camera_input("Camera Feed", key="camera")
        if img_file_buffer is None:
            status_placeholder.error("⚠️ No camera frame available.")
            time.sleep(0.1)
            continue

        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize and normalize
        h0, w0 = frame.shape[:2]
        scale = process_width / float(w0)
        frame = cv2.resize(frame, (process_width, int(h0 * scale)))
        h, w = frame.shape[:2]
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        command = None

        # Detection logic: hands, face, blink, head
        hand_result = hands.process(rgb) if enable_hand else None
        face_result = face_mesh.process(rgb) if (enable_eye or enable_head) else None

        # --- Hand Detection ---
        if enable_hand and hand_result and hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand_landmarks)
                g = detect_hand_gesture(fingers)
                if g and g != last_gesture:
                    last_gesture = g
                    gesture_last_time = time.time()
                    enqueue_audio(GESTURE_SOUNDS.get(g, None))
                    command = g

        # --- Face/Eye/Head Detection ---
        if (enable_eye or enable_head) and face_result and face_result.multi_face_landmarks:
            landmarks = face_result.multi_face_landmarks[0].landmark

            # Blink detection
            if enable_eye:
                ear = eye_aspect_ratio(landmarks, LEFT_EYE, RIGHT_EYE)
                if ear < EAR_THRESHOLD:
                    blink_frame_counter += 1
                else:
                    if blink_frame_counter >= CONSEC_FRAMES:
                        duration = blink_frame_counter / 1.0
                        now = time.time()
                        if duration >= LONG_BLINK_SECONDS:
                            long_blink_events += 1
                            total_blinks += 1
                            last_blink_event = "Long Blink"
                            last_blink_event_time = now
                            enqueue_audio(SINGLE_BLINK_AUDIO)
                        else:
                            if pending_blink_time is None:
                                pending_blink_time = now
                            else:
                                if now - pending_blink_time <= DOUBLE_BLINK_MAX_INTERVAL:
                                    double_blink_events += 1
                                    total_blinks += 2
                                    last_blink_event = "Double Blink"
                                    last_blink_event_time = now
                                    pending_blink_time = None
                                    enqueue_audio(DOUBLE_BLINK_AUDIO)
                                else:
                                    single_blink_events += 1
                                    total_blinks += 1
                                    last_blink_event = "Single Blink"
                                    last_blink_event_time = now
                                    pending_blink_time = now
                                    enqueue_audio(SINGLE_BLINK_AUDIO)
                    blink_frame_counter = 0

            # Head pose detection
            if enable_head:
                yaw, pitch, roll = get_head_pose(landmarks, w, h)
                if yaw is not None:
                    yaw_q.append(yaw)
                    pitch_q.append(pitch)
                    smooth_yaw = float(np.median(list(yaw_q)))
                    smooth_pitch = float(np.median(list(pitch_q)))
                    detected_dir = get_direction_from_pose(smooth_yaw, smooth_pitch)
                    if detected_dir != head_last_detected:
                        head_last_detected = detected_dir
                        head_detect_time = time.time()
                    else:
                        if (time.time() - head_detect_time) > STABLE_TIME:
                            if head_stable != detected_dir:
                                head_stable = detected_dir
                                enqueue_audio(AUDIO_PATHS.get(head_stable, None))
                    cv2.putText(frame, f"Yaw:{int(smooth_yaw)} Pitch:{int(smooth_pitch)}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, f"HeadDir:{head_stable}", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Display command/blink
        y_text = 100
        if command:
            cv2.putText(frame, f"{command}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255),3)
            y_text += 40
        if last_blink_event and (time.time() - last_blink_event_time) < 2.0:
            cv2.putText(frame, last_blink_event, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0),3)
            y_text += 30

        cv2.putText(frame, f"Total blinks: {total_blinks}", (10,h-50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)
        cv2.putText(frame, f"Singles:{single_blink_events} Doubles:{double_blink_events} Longs:{long_blink_events}",
                    (10,h-25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),2)

        # FPS
        frame_times.append(time.time()-loop_start)
        fps = 1.0 / (sum(frame_times)/len(frame_times)) if frame_times else 0.0
        fps_placeholder.metric("FPS", f"{fps:.1f}")

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elapsed = time.time() - loop_start
        if elapsed < 0.03:
            time.sleep(0.03 - elapsed)

except Exception as e:
    st.error(f"Error in main loop: {e}")

finally:
    stop_audio()
    if audio_thread:
        audio_queue.put(None)
        audio_thread.join(timeout=0.5)
    st.info("Stopped")
