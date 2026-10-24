"""
game_control.py 

Captures webcam, runs a smart Hand Presence Gate (MOG2 background modeling)
to prevent idle background false-triggers, and translates active hand gestures
into DirectInput hardware keystrokes (Arrow Keys + WASD) for Subway Surfers and Cyber Surfer.

Action Mappings:
    JUMP   : 03_fist, 04_fist_moved        -> Arrow Up + W
    ROLL   : 01_palm, 08_palm_moved, 10_down -> Arrow Down + S
    LEFT   : 02_l, 09_c                    -> Arrow Left + A
    RIGHT  : 05_thumb, 06_index            -> Arrow Right + D
    IDLE   : No hand raised / at rest       -> NO KEYS SENT

Controls in OpenCV Window:
    SPACE   - Start / Pause transmission
    C       - Recalibrate background (sit naturally and press C)
    + / =   - Increase confidence threshold (+0.05)
    - / _   - Decrease confidence threshold (-0.05)
    [ / ]   - Decrease / Increase cooldown duration (0.05s)
    Q / ESC - Quit
"""

import os
import sys
import time
import json
import ctypes
from ctypes import wintypes
from collections import deque

import numpy as np
import cv2
import tensorflow as tf
from pynput.keyboard import Controller, Key

# --- Windows DirectInput Hardware Key Simulator ---
user32 = ctypes.windll.user32

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ('wVk', wintypes.WORD),
        ('wScan', wintypes.WORD),
        ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.c_ulonglong)
    ]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [('ki', KEYBDINPUT)]
    _anonymous_ = ('_input',)
    _fields_ = [('type', wintypes.DWORD), ('_input', _INPUT)]

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

SCAN_UP = 0x48       # Arrow Up (requires extended flag)
SCAN_DOWN = 0x50     # Arrow Down (requires extended flag)
SCAN_LEFT = 0x4B     # Arrow Left (requires extended flag)
SCAN_RIGHT = 0x4D    # Arrow Right (requires extended flag)
SCAN_W = 0x11        # 'W' key
SCAN_S = 0x1F        # 'S' key
SCAN_A = 0x1E        # 'A' key
SCAN_D = 0x20        # 'D' key


def press_hardware_key(scan_code, extended=False, duration=0.04):
    """Sends low-level hardware scan code directly to Windows input queue."""
    flags_down = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0)
    flags_up = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP | (KEYEVENTF_EXTENDEDKEY if extended else 0)

    inp_down = INPUT(type=1, ki=KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=flags_down, time=0, dwExtraInfo=0))
    user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))
    time.sleep(duration)
    inp_up = INPUT(type=1, ki=KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=flags_up, time=0, dwExtraInfo=0))
    user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))


# --- Configuration ---
IMG_SIZE = 64
MODEL_PATHS = [
    "gesture_cnn_finetuned.keras",
    "gesture_cnn_finetuned.h5",
    "gesture_cnn_final.keras",
    "gesture_cnn_final.h5",
]
LABEL_MAP_PATH = "label_map.json"

DEFAULT_CONFIDENCE = float(os.getenv("GESTURE_CONFIDENCE", "0.55"))
DEFAULT_COOLDOWN = 0.35
SMOOTHING_WINDOW = 2
MIN_HAND_MOTION_RATIO = 1.4  # % of pixels in hand zone that must be active to trigger

# Gesture classes mapped to Action Groups
GESTURE_TO_ACTION = {
    "03_fist": "JUMP",
    "04_fist_moved": "JUMP",
    "01_palm": "ROLL",
    "08_palm_moved": "ROLL",
    "10_down": "ROLL",
    "02_l": "LEFT",
    "09_c": "LEFT",
    "05_thumb": "RIGHT",
    "06_index": "RIGHT",
}

ACTION_DISPLAY = {
    "JUMP": "JUMP (UP / W)",
    "ROLL": "ROLL / SLIDE (DOWN / S)",
    "LEFT": "MOVE LEFT (LEFT / A)",
    "RIGHT": "MOVE RIGHT (RIGHT / D)",
}


def load_model_and_labels():
    model = None
    chosen_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                print(f"Loading model: {path} ...")
                model = tf.keras.models.load_model(path)
                chosen_path = path
                print(f"Loaded: {path}")
                break
            except Exception as e:
                print(f"Could not load {path}: {e}")

    if model is None:
        raise FileNotFoundError(f"Could not load model from: {MODEL_PATHS}")

    with open(LABEL_MAP_PATH, "r") as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}
    return model, idx_to_label, chosen_path


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return normalized[np.newaxis, ..., np.newaxis]


def send_game_action(action_name, keyboard):
    """
    Sends DirectInput hardware scan codes (both Arrow and WASD)
    plus pynput keys to guarantee compatibility with Subway Surfers (Unity WebGL).
    """
    if action_name == "JUMP":
        press_hardware_key(SCAN_UP, extended=True)
        press_hardware_key(SCAN_W, extended=False)
        try:
            keyboard.press(Key.up)
            keyboard.press('w')
            time.sleep(0.04)
            keyboard.release(Key.up)
            keyboard.release('w')
        except Exception:
            pass

    elif action_name == "ROLL":
        press_hardware_key(SCAN_DOWN, extended=True)
        press_hardware_key(SCAN_S, extended=False)
        try:
            keyboard.press(Key.down)
            keyboard.press('s')
            time.sleep(0.04)
            keyboard.release(Key.down)
            keyboard.release('s')
        except Exception:
            pass

    elif action_name == "LEFT":
        press_hardware_key(SCAN_LEFT, extended=True)
        press_hardware_key(SCAN_A, extended=False)
        try:
            keyboard.press(Key.left)
            keyboard.press('a')
            time.sleep(0.04)
            keyboard.release(Key.left)
            keyboard.release('a')
        except Exception:
            pass

    elif action_name == "RIGHT":
        press_hardware_key(SCAN_RIGHT, extended=True)
        press_hardware_key(SCAN_D, extended=False)
        try:
            keyboard.press(Key.right)
            keyboard.press('d')
            time.sleep(0.04)
            keyboard.release(Key.right)
            keyboard.release('d')
        except Exception:
            pass


def draw_hud(frame, running, hand_active, gesture, action, confidence, threshold, cooldown, fps, last_action, action_flash_time, is_calibrating):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top banner
    cv2.rectangle(overlay, (0, 0), (w, 82), (18, 18, 24), -1)
    # Bottom banner
    cv2.rectangle(overlay, (0, h - 75), (w, h), (18, 18, 24), -1)
    # Right cheat-sheet card
    cheat_w = 240
    cv2.rectangle(overlay, (w - cheat_w - 10, 92), (w - 10, 240), (22, 22, 28), -1)

    alpha = 0.76
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Status Pill
    if is_calibrating:
        state_text = "CALIBRATING BACKGROUND..."
        state_bg = (0, 165, 255)
    elif not running:
        state_text = "PAUSED (SPACE to resume)"
        state_bg = (40, 130, 240)
    elif hand_active:
        state_text = "HAND ACTIVE (GESTURING)"
        state_bg = (35, 195, 55)
    else:
        state_text = "IDLE (WAITING FOR HAND)"
        state_bg = (80, 80, 95)

    cv2.rectangle(frame, (12, 12), (235, 42), state_bg, -1, cv2.LINE_AA)
    cv2.putText(frame, state_text, (20, 33), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Top instructions
    cv2.putText(frame, "Click Subway Surfers to focus! | C: Recalibrate | SPACE: Pause | Q: Quit",
                (250, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 215, 215), 1, cv2.LINE_AA)

    # Stats line
    cv2.putText(frame, f"FPS: {fps:.1f} | Threshold: {threshold:.2f} (+/- adjust) | Cooldown: {cooldown:.2f}s",
                (250, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (160, 230, 160), 1, cv2.LINE_AA)

    # Bottom Banner: Gesture Detection & Bar
    if hand_active:
        detected_color = (0, 255, 120) if confidence >= threshold else (80, 140, 255)
        action_info = ACTION_DISPLAY.get(action, "IDENTIFYING...")
        cv2.putText(frame, f"Gesture: {gesture}  ->  {action_info}", (15, h - 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

        bar_x, bar_y, bar_w, bar_h = 15, h - 28, 280, 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1, cv2.LINE_AA)
        filled_w = int(bar_w * max(0.0, min(1.0, confidence)))
        cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + filled_w, bar_y + bar_h - 1), detected_color, -1)

        thresh_x = int(bar_x + bar_w * threshold)
        cv2.line(frame, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (255, 255, 255), 2)
        cv2.putText(frame, f"{confidence * 100:.1f}%", (bar_x + bar_w + 10, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, detected_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Status: IDLE / NEUTRAL  (Raise hand in box to gesture)", (15, h - 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, (180, 180, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, "No keys being sent. Character running straight.", (15, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 150), 1, cv2.LINE_AA)

    # Right Cheat Sheet
    cv2.putText(frame, "SUBWAY SURFERS CONTROLS", (w - cheat_w, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
    items = [
        ("Fist (03/04)", "JUMP  (UP/W)"),
        ("Palm (01/08)", "ROLL  (DOWN/S)"),
        ("L-Sign (02)", "LEFT  (LEFT/A)"),
        ("Thumb (05)", "RIGHT (RIGHT/D)"),
    ]
    for i, (g_key, act) in enumerate(items):
        y_pos = 138 + i * 24
        is_current_action = (hand_active and action == ["JUMP", "ROLL", "LEFT", "RIGHT"][i] and confidence >= threshold)
        color = (0, 255, 120) if is_current_action else (175, 175, 175)
        prefix = ">" if is_current_action else "-"
        cv2.putText(frame, f"{prefix} {g_key}: {act}", (w - cheat_w, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    # Hand Placement Guide Box
    box_sz = int(min(h, w) * 0.52)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - box_sz // 2, cy - box_sz // 2
    x2, y2 = cx + box_sz // 2, cy + box_sz // 2
    guide_col = (0, 255, 120) if (hand_active and confidence >= threshold) else (100, 100, 115)
    c_len = 24
    for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        dx = c_len if px == x1 else -c_len
        dy = c_len if py == y1 else -c_len
        cv2.line(frame, (px, py), (px + dx, py), guide_col, 2)
        cv2.line(frame, (px, py), (px, py + dy), guide_col, 2)

    # Action Trigger Flash Banner
    now = time.time()
    if last_action and (now - action_flash_time < 0.45):
        banner_w, banner_h = 340, 48
        bx1 = (w - banner_w) // 2
        by1 = 92
        cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, f"KEY SENT: {last_action}!", (bx1 + 20, by1 + 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (10, 10, 15), 2, cv2.LINE_AA)


def main():
    model, idx_to_label, model_path = load_model_and_labels()
    keyboard = Controller()

    confidence_threshold = DEFAULT_CONFIDENCE
    cooldown_seconds = DEFAULT_COOLDOWN

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Use 1280x720 16:9
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize MOG2 background subtractor to detect hand presence
    fgbg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=25, detectShadows=False)

    print("\n" + "=" * 65)
    print("  SUBWAY SURFERS GESTURE CONTROLLER INITIALIZING")
    print(f"  Model: {model_path}")
    print("  Calibrating ambient background for 1 second... Please sit naturally.")
    print("=" * 65 + "\n")

    # Warmup and calibrate background on startup
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            fgbg.apply(frame, learningRate=0.08)

    print("  CALIBRATION COMPLETE: Ready! Show hand gestures to play.")
    print("  IMPORTANT: Click on Subway Surfers on Poki to focus the game.\n")

    action_history = deque(maxlen=SMOOTHING_WINDOW)
    last_action_time = 0
    last_action_label = ""
    action_flash_time = 0
    running = True

    fps_frame_times = deque(maxlen=30)
    is_calibrating = False
    calib_frames_left = 0

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame.")
            break
        frame = cv2.flip(frame, 1)

        # Handle recalibration
        if is_calibrating:
            fgbg.apply(frame, learningRate=0.15)
            calib_frames_left -= 1
            if calib_frames_left <= 0:
                is_calibrating = False
                print("Recalibration complete!")

        # Hand presence check via background motion gate
        h, w = frame.shape[:2]
        box_sz = int(min(h, w) * 0.52)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - box_sz // 2, cy - box_sz // 2
        x2, y2 = cx + box_sz // 2, cy + box_sz // 2
        roi = frame[y1:y2, x1:x2]

        fgmask = fgbg.apply(frame, learningRate=0.003 if not is_calibrating else 0.15)
        roi_fgmask = fgmask[y1:y2, x1:x2]

        fg_ratio = np.mean(roi_fgmask > 0) * 100
        hand_active = (fg_ratio >= MIN_HAND_MOTION_RATIO) and not is_calibrating

        gesture = "neutral"
        action = None
        confidence = 0.0

        if hand_active:
            input_tensor = preprocess(frame)
            preds = model.predict(input_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx])
            gesture = idx_to_label.get(pred_idx, "unknown")
            action = GESTURE_TO_ACTION.get(gesture, None)

            # Special filter: for 02_l (LEFT), require higher confidence (>=0.70)
            # to prevent any lingering background bias
            if action == "LEFT" and confidence < 0.70:
                action = None

            confident = confidence >= confidence_threshold
            action_history.append(action if confident else None)
        else:
            action_history.append(None)

        # Temporal smoothing by action group
        smoothed_action = None
        if len(action_history) == SMOOTHING_WINDOW and len(set(action_history)) == 1 and action_history[0] is not None:
            smoothed_action = action_history[0]

        now = time.time()
        if running and smoothed_action is not None and hand_active:
            if now - last_action_time >= cooldown_seconds:
                display_label = ACTION_DISPLAY.get(smoothed_action, smoothed_action)
                send_game_action(smoothed_action, keyboard)
                last_action_time = now
                last_action_label = smoothed_action
                action_flash_time = now
                print(f">>> [KEY SENT TO GAME] {display_label}! (Gesture: {gesture}, Conf: {confidence*100:.1f}%)", flush=True)

        fps_frame_times.append(time.time() - frame_start)
        current_fps = 1.0 / (sum(fps_frame_times) / len(fps_frame_times)) if fps_frame_times else 0.0

        draw_hud(frame, running, hand_active, gesture, action, confidence, confidence_threshold,
                 cooldown_seconds, current_fps, last_action_label, action_flash_time, is_calibrating)

        cv2.imshow("Subway Surfers Gesture Controller", frame)

        key_press = cv2.waitKey(1) & 0xFF
        if key_press in (ord('q'), ord('Q'), 27):
            break
        elif key_press == ord(' '):
            running = not running
            print(f"Controller {'ACTIVE' if running else 'PAUSED'}.")
        elif key_press in (ord('c'), ord('C')):
            is_calibrating = True
            calib_frames_left = 30
            print("Recalibrating background... Sit naturally.")
        elif key_press in (ord('+'), ord('=')):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")
        elif key_press in (ord('-'), ord('_')):
            confidence_threshold = max(0.25, confidence_threshold - 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")
        elif key_press == ord(']'):
            cooldown_seconds = min(1.5, cooldown_seconds + 0.05)
            print(f"Cooldown: {cooldown_seconds:.2f}s")
        elif key_press == ord('['):
            cooldown_seconds = max(0.15, cooldown_seconds - 0.05)
            print(f"Cooldown: {cooldown_seconds:.2f}s")

    cap.release()
    cv2.destroyAllWindows()
    print("Subway Surfers Gesture Controller closed cleanly.")


if __name__ == "__main__":
    main()
