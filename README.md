# 🎮 GestureGame: Real-Time Hand Gesture Controlled Endless Runner

Touchless, real-time gesture recognition using Deep Convolutional Neural Networks (CNN), OpenCV, and TensorFlow to control arcade endless runner games hands-free.

---

## 🚀 Features

- **Fine-Tuned CNN Model**: Trained on LeapGestRecog and fine-tuned locally on real webcam data across multiple lighting conditions and angles (`gesture_cnn_finetuned.keras`).
- **Real-Time Gesture Pipeline**: Preprocessing $\rightarrow$ CNN inference $\rightarrow$ Confidence filtering $\rightarrow$ Temporal smoothing $\rightarrow$ Debounced keystroke emulation.
- **Cyber Surfer Game**: Built-in 3D synthwave endless runner with Web Audio synth sound effects, 3 lanes, jumping over laser barriers, sliding under overhead beams, and collecting coins.
- **Universal Compatibility**: Works out of the box with the built-in game as well as external games like [Subway Surfers on Poki](https://poki.com/en/g/subway-surfers).
- **Interactive OpenCV HUD**: Real-time FPS counter, confidence progress bar, live gesture status, hand placement guide, and keyboard shortcuts to tune thresholds on the fly.

---

## ✋ Gesture Mapping

| Gesture | Label Map | Keypress | Game Action |
| :--- | :--- | :--- | :--- |
| ✊ **Fist** | `03_fist` | `Up Arrow` (`↑`) | **Jump** (hurdles, trains) |
| ✋ **Palm** | `01_palm` | `Down Arrow` (`↓`) | **Slide / Duck** (overhead beams) |
| 👈 **L-Sign** | `02_l` | `Left Arrow` (`←`) | **Move Left** |
| 👍 **Thumb** | `05_thumb` | `Right Arrow` (`→`) | **Move Right** |

---

## ⚡ Quick Start

### 1. Unified Launcher (Recommended)
Starts the local game server, opens the browser, and launches the webcam gesture recognizer automatically:

```bash
.\venv\Scripts\python run_game.py
```

### 2. Standalone Gesture Control
If you already have your own game open (e.g. Subway Surfers in Chrome):

```bash
.\venv\Scripts\python game_control.py
```

---

## ⌨️ Webcam Window Controls

While the OpenCV webcam window is focused:
- **`SPACE`**: Start / Pause gesture transmission (starts paused so you can focus your game window).
- **`P`**: Pause keypress transmission.
- **`+` / `=`**: Increase confidence threshold (+0.05).
- **`-` / `_`**: Decrease confidence threshold (-0.05).
- **`[` / `]`**: Adjust keypress cooldown duration.
- **`Q` or `ESC`**: Quit the application.

---

## 📂 Project Structure

```
GestureGame/
├── run_game.py                 # Unified launcher (browser game + webcam controller)
├── game_control.py             # Phase 8: Real-time webcam inference & key simulation
├── finetune_local.py           # Phase 7: Transfer learning / fine-tuning on webcam data
├── collect_webcam_data.py      # Phase 6: Webcam dataset gathering tool
├── game.html                   # Cyber Surfer 3D synthwave runner game
├── gesture_cnn_finetuned.keras # Fine-tuned CNN model (90.7% train, 63.2% held-out test)
├── gesture_cnn_final.keras     # Base trained CNN model (Keras format)
├── gesture_cnn_final.h5        # Base trained CNN model (HDF5 format)
├── label_map.json              # Class index to gesture name mapping
├── requirements.txt            # Python dependencies
└── webcam_data/                # Captured local webcam training dataset (800+ images)
```
