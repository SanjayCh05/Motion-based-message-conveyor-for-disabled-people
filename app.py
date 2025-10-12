import os
import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Suppress ALSA and other audio warnings
os.environ["ALSA_LOG_LEVEL"] = "0"

# =========================================================
# 0. STREAMLIT CONFIG (must be first)
# =========================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Unified Caregiver Instruction System", layout="wide")
    st.title("🧠 Unified Real-time Detection System for Caregiver Instructions (No Audio)")
    st.markdown("""
    This system uses **Hand Gestures**, **Head Pose**, and **Blink Patterns** for non-verbal communication.

    ---
    **Detection Status:**
    * **Hand Gestures:** Standard hand signs (e.g., index finger up).
    * **Head Pose:** Stabilized detection after holding a direction for **0.5 seconds**.
    * **Blinks:** Commands are triggered by a **Double Blink**.
    """)

# =========================================================
# 1. MEDIAPIPE & CONSTANTS SETUP
# =========================================================
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0), (0.0, -330.0, -65.0)
])
FACE_LANDMARK_POINTS = [1, 33, 263, 61, 291, 152]
HEAD_YAW_THRESHOLD = 10
HEAD_PITCH_THRESHOLD = 12
HEAD_STABLE_TIME = 0.5

LEFT_EYE_EAR_INDICES = [362, 380, 374, 263, 386, 385]
RIGHT_EYE_EAR_INDICES = [33, 159, 158, 133, 153, 145]
EAR_THRESHOLD = 0.3
CONSEC_FRAMES = 4
DOUBLE_BLINK_MAX_INTERVAL = 0.8

FINGER_TIPS = [8, 12, 16, 20]

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def fingers_up(hand_landmarks):
    fingers = [1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0]
    for tip in FINGER_TIPS:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)
    return fingers

def detect_gesture(fingers):
    gestures = {
        (0, 1, 0, 0, 0): "Bring Water", (0, 1, 1, 0, 0): "Emergency", (0, 0, 0, 0, 0): "Stop",
        (0, 1, 1, 1, 0): "Assist me outside", (0, 1, 1, 1, 1): "Call 108",
        (0, 0, 1, 1, 1): "Contact my caregiver", (0, 0, 0, 0, 1): "Check my supplies",
        (0, 0, 0, 1, 1): "Help me sit"
    }
    return gestures.get(tuple(fingers))

def get_head_pose(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in FACE_LANDMARK_POINTS
    ], dtype="double")
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )
    if not success:
        return None, None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, _ = eulerAngles.flatten()[:3]
    return yaw, pitch

def get_head_direction(yaw, pitch):
    if yaw > HEAD_YAW_THRESHOLD:
        return "Right"
    elif yaw < -HEAD_YAW_THRESHOLD:
        return "Left"
    elif pitch > HEAD_PITCH_THRESHOLD:
        return "Up"
    elif pitch < -HEAD_PITCH_THRESHOLD:
        return "Down"
    else:
        return "Center"

def get_head_instruction(direction):
    if direction == "Up": return "WATER"
    if direction == "Down": return "EMERGENCY"
    if direction == "Left": return "FIRST AID"
    if direction == "Right": return "FOOD"
    return "Looking straight"

def eye_aspect_ratio(eye_landmarks_indices, face_landmarks):
    def get_coords(idx): return np.array([face_landmarks[idx].x, face_landmarks[idx].y])
    A = np.linalg.norm(get_coords(eye_landmarks_indices[1]) - get_coords(eye_landmarks_indices[5]))
    B = np.linalg.norm(get_coords(eye_landmarks_indices[2]) - get_coords(eye_landmarks_indices[4]))
    C = np.linalg.norm(get_coords(eye_landmarks_indices[0]) - get_coords(eye_landmarks_indices[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0.0

# =========================================================
# 3. VIDEO TRANSFORMER CLASS (Unified Logic)
# =========================================================
class UnifiedVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6)
        self.stable_direction = "Center"
        self.last_detected_direction = "Center"
        self.direction_start_time = time.time()
        self.blink_counter = 0
        self.double_blink_counter = 0
        self.last_event = None
        self.last_event_time = 0.0
        self.last_gesture = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_h, img_w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Hand Gesture Detection ---
        hand_result = self.hands.process(rgb)
        current_gesture = None
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand_landmarks)
                current_gesture = detect_gesture(fingers)

        if current_gesture and current_gesture != self.last_gesture:
            self.last_gesture = current_gesture
        elif not current_gesture:
            self.last_gesture = None

        if current_gesture:
            cv2.putText(img, f"Hand: {current_gesture}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # --- Face Mesh, Head Pose, Blink Detection ---
        face_result = self.face_mesh.process(rgb)

        if face_result.multi_face_landmarks:
            landmarks = face_result.multi_face_landmarks[0].landmark
            mp_draw.draw_landmarks(img, face_result.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)

            # Head Pose
            yaw, pitch = get_head_pose(landmarks, img_w, img_h)
            if yaw is not None:
                detected_direction = get_head_direction(yaw, pitch)
                current_time = time.time()
                if detected_direction != self.last_detected_direction:
                    self.last_detected_direction = detected_direction
                    self.direction_start_time = current_time
                else:
                    if current_time - self.direction_start_time > HEAD_STABLE_TIME:
                        self.stable_direction = detected_direction

                instruction_text = get_head_instruction(self.stable_direction)
                cv2.putText(img, f"Head: {instruction_text}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Blink Detection
            right_ear = eye_aspect_ratio(RIGHT_EYE_EAR_INDICES, landmarks)
            left_ear = eye_aspect_ratio(LEFT_EYE_EAR_INDICES, landmarks)
            avg_ear = (right_ear + left_ear) / 2.0

            event_detected = None

            if avg_ear < EAR_THRESHOLD:
                self.blink_counter += 1
            else:
                if self.blink_counter >= CONSEC_FRAMES:
                    current_time_for_blink = time.time()
                    if self.last_event == "Single Blink" and (current_time_for_blink - self.last_event_time) < DOUBLE_BLINK_MAX_INTERVAL:
                        self.double_blink_counter += 1
                        event_detected = "Double Blink (COMMAND)"
                        self.last_event = None
                    elif self.last_event is None:
                        event_detected = "Single Blink"
                        self.last_event = event_detected
                        self.last_event_time = current_time_for_blink
                    self.blink_counter = 0
                elif self.last_event == "Single Blink" and (time.time() - self.last_event_time) >= DOUBLE_BLINK_MAX_INTERVAL:
                    self.last_event = None
                self.blink_counter = 0

            cv2.putText(img, f"Double Blinks: {self.double_blink_counter}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if self.last_event:
                cv2.putText(img, f"Last Event: {self.last_event}", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

# =========================================================
# 4. STREAMLIT WEBRTC CALL
# =========================================================
webrtc_streamer(
    key="unified_detection_system",
    video_processor_factory=UnifiedVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
