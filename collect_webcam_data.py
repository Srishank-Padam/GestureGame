"""
collect_webcam_data.py  (Phase 6: Webcam Dataset Collection)

Run this LOCALLY (not in Colab) to collect your own webcam gesture images.
LeapGestRecog was captured with an IR camera under lab conditions; a normal webcam
under your actual demo lighting looks different, so the CNN needs real examples of
that to generalize (Phase 7 fine-tuning depends on this data).

Usage:
    python collect_webcam_data.py

Controls (shown on screen):
    - Press the number key next to a gesture to start/stop capturing for it.
      Keys 1-9 map to classes 1-9, and key 0 maps to the 10th class, matching the
      on-screen numbering exactly (no more mismatch between the key you press and
      the number shown).
    - Captures ~2 images/second while "recording" for a class.
    - Press 'q' to quit.

Guided capture: for each gesture, you'll be walked through four conditions in turn
(near / far / rotated / different lighting) so the fine-tuning set has real variation
instead of 80 near-identical frames from one pose.

Output:
    webcam_data/<gesture_name>/<condition>/img_0001.png, img_0002.png, ...
"""

import cv2
import os
import time

# Must match the class names/order used during training (see label_map.json from Colab)
GESTURE_CLASSES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]

CONDITIONS = ["near", "far", "rotated", "different_light"]
IMAGES_PER_CONDITION = 20  # 4 conditions x 20 = 80 per class, matching TARGET_PER_CLASS

OUTPUT_DIR = "webcam_data"
CAPTURE_INTERVAL = 0.5  # seconds between captures while recording

# Key -> class index. Key '1' is class 0, ..., key '9' is class 8, key '0' is class 9.
# This makes the on-screen [1]..[9],[0] labels match exactly what you press.
def key_to_class_index(key_char):
    if key_char == '0':
        return 9
    if key_char.isdigit() and key_char != '0':
        idx = int(key_char) - 1
        if 0 <= idx < len(GESTURE_CLASSES):
            return idx
    return None

def display_key_for_index(i):
    return '0' if i == 9 else str(i + 1)


def main():
    for g in GESTURE_CLASSES:
        for c in CONDITIONS:
            os.makedirs(os.path.join(OUTPUT_DIR, g, c), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permissions/index.")
        return

    recording_class = None
    current_condition_idx = {g: 0 for g in GESTURE_CLASSES}  # which condition we're on, per class
    last_capture_time = 0

    def counts_for(g):
        return {c: len(os.listdir(os.path.join(OUTPUT_DIR, g, c))) for c in CONDITIONS}

    print("Press the number key next to a gesture to start/stop recording it.")
    print("Each gesture cycles through: near -> far -> rotated -> different_light")
    print("Press 'q' to quit.\n")
    for i, g in enumerate(GESTURE_CLASSES):
        print(f"  [{display_key_for_index(i)}] {g}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        if recording_class:
            cond = CONDITIONS[current_condition_idx[recording_class]]
            status = f"Recording: {recording_class} [{cond}]"
        else:
            status = "Not recording"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if recording_class else (0, 0, 255), 2)

        y = 60
        for i, g in enumerate(GESTURE_CLASSES):
            counts = counts_for(g)
            total = sum(counts.values())
            key_label = display_key_for_index(i)
            cv2.putText(display, f"[{key_label}] {g}: {total}/{len(CONDITIONS)*IMAGES_PER_CONDITION}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20

        cv2.imshow("Webcam Data Collection", display)

        if recording_class is not None and time.time() - last_capture_time > CAPTURE_INTERVAL:
            cond = CONDITIONS[current_condition_idx[recording_class]]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            existing = len(os.listdir(os.path.join(OUTPUT_DIR, recording_class, cond)))
            idx = existing + 1
            fpath = os.path.join(OUTPUT_DIR, recording_class, cond, f"img_{idx:04d}.png")
            cv2.imwrite(fpath, gray)
            last_capture_time = time.time()

            if idx >= IMAGES_PER_CONDITION:
                if current_condition_idx[recording_class] < len(CONDITIONS) - 1:
                    current_condition_idx[recording_class] += 1
                    next_cond = CONDITIONS[current_condition_idx[recording_class]]
                    print(f"{recording_class}: '{cond}' done. Reposition for '{next_cond}', still recording.")
                else:
                    print(f"{recording_class}: all conditions complete, stopping.")
                    recording_class = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        else:
            try:
                key_char = chr(key)
            except ValueError:
                key_char = None
            if key_char:
                idx = key_to_class_index(key_char)
                if idx is not None:
                    chosen = GESTURE_CLASSES[idx]
                    recording_class = None if recording_class == chosen else chosen

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal counts:")
    for g in GESTURE_CLASSES:
        counts = counts_for(g)
        print(f"  {g}: {sum(counts.values())} total  ({counts})")

if __name__ == "__main__":
    main()
