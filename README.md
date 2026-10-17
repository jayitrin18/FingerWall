# FingerWall ✍️

A webcam-based virtual writing board that lets you draw and write in the air using hand gestures.

## Features

- Real-time hand tracking using a webcam
- Freehand drawing with your index finger
- Multiple pen colours and adjustable thickness
- Gesture-based erasing
- Live webcam preview
- Save drawings as PNG images
- Full-screen black canvas

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- PyInstaller

## How It Works

FingerWall uses webcam input and hand-landmark detection to track finger movements and translate them into strokes on a virtual canvas.

## Controls

| Action | Control |
|---|---|
| Draw | Raise index finger and curl middle finger |
| Erase with gesture | Raise index and middle fingers, curl ring finger |
| Select colour | Keys 1–5 |
| Increase pen thickness | `+` |
| Decrease pen thickness | `-` |
| Toggle eraser mode | `E` |
| Toggle camera preview | `V` |
| Clear canvas | `C` |
| Save drawing | `S` |
| Exit | `Q` |

## Run from Source

1. Install Python.
2. Install the required packages: `mediapipe`, `opencv-python`, and `numpy`.
3. Download `hand_landmarker.task` into the project folder.
4. Run:

```bash
python fingerwall.py
```

## Windows Desktop App

Download the Windows ZIP from the repository's **Releases** section, extract it, and launch `FingerWall.exe`.

Keep the `_internal` folder alongside the executable.

## Author

Created as a Python computer-vision project.

