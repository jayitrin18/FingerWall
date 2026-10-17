
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math

# ---------- SETTINGS ----------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "hand_landmarker.task")

CAM_W, CAM_H = 640, 480
W, H = 1280, 720

COLORS = [
    ("Cyan", (255, 255, 0)),
    ("Pink", (255, 0, 255)),
    ("Yellow", (0, 255, 255)),
    ("Green", (0, 255, 0)),
    ("White", (255, 255, 255)),
]

color_index = 0
thickness = 5
eraser_size = 35
eraser_mode = False
show_camera = True

canvas = np.zeros((H, W, 3), dtype=np.uint8)
previous = None
smooth_point = None
drawing = False
timestamp = 0
last_saved = ""

# ---------- MEDIAPIPE ----------
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

if not camera.isOpened():
    raise RuntimeError(
        "Camera could not open. Close other apps using your webcam."
    )

window = "FingerWall"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
)

def draw_text(img, text, pos, scale=0.6,
              color=(230, 230, 230), thick=1):
    cv2.putText(
        img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thick, cv2.LINE_AA
    )

def draw_segment(img, p1, p2, color, width):
    distance = math.dist(p1, p2)

    # Ignore very large tracking jumps.
    if distance > 120:
        return

    cv2.line(
        img, p1, p2, color, width, cv2.LINE_AA
    )

def save_board():
    filename = os.path.join(
        BASE, "FingerWall_" +
        time.strftime("%Y%m%d_%H%M%S") + ".png"
    )

    if cv2.imwrite(filename, canvas):
        print("Saved:", filename)
        return filename

    print("Could not save the image.")
    return ""

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as tracker:

        print("\nFINGERWALL READY")
        print("Index up + middle curled: DRAW")
        print("Index + middle up + ring curled: ERASE")
        print("1-5: Colours")
        print("+ / -: Pen thickness")
        print("E: Toggle eraser | V: Camera preview")
        print("C: Clear | S: Save PNG | Q: Quit\n")

        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAM_W, CAM_H))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp += 33
            result = tracker.detect_for_video(image, timestamp)

            hand_detected = False
            gesture = "READY"
            tip_position = None
            erase_gesture = False
            draw_gesture = False

            if result.hand_landmarks:
                hand_detected = True
                hand = result.hand_landmarks[0]

                # Fingertip and joint landmarks
                index_tip = hand[8]
                index_joint = hand[6]
                middle_tip = hand[12]
                middle_joint = hand[10]
                ring_tip = hand[16]
                ring_joint = hand[14]

                index_up = (
                    index_tip.y < index_joint.y - 0.015
                )
                middle_up = (
                    middle_tip.y < middle_joint.y - 0.015
                )
                ring_curled = (
                    ring_tip.y > ring_joint.y + 0.015
                )
                middle_curled = (
                    middle_tip.y > middle_joint.y + 0.015
                )

                # Draw: index finger up, middle finger curled.
                draw_gesture = index_up and middle_curled

                # Erase: index and middle up, ring finger curled.
                erase_gesture = (
                    index_up and middle_up and ring_curled
                )

                x = int(index_tip.x * W)
                y = int(index_tip.y * H)
                tip_position = (
                    max(0, min(W - 1, x)),
                    max(0, min(H - 1, y))
                )

                if erase_gesture:
                    gesture = "ERASING"
                    drawing = False
                    previous = None
                    smooth_point = None

                    ex, ey = tip_position
                    cv2.circle(
                        canvas, (ex, ey), eraser_size,
                        (0, 0, 0), -1, cv2.LINE_AA
                    )

                elif draw_gesture:
                    gesture = (
                        "ERASER MODE" if eraser_mode
                        else "DRAWING"
                    )

                    point = np.array(
                        tip_position, dtype=float
                    )

                    if smooth_point is None or not drawing:
                        smooth_point = point
                        previous = tuple(
                            np.round(point).astype(int)
                        )
                    else:
                        smooth_point = (
                            0.65 * smooth_point + 0.35 * point
                        )

                    current = tuple(
                        np.round(smooth_point).astype(int)
                    )

                    if previous is not None:
                        if eraser_mode:
                            draw_segment(
                                canvas, previous, current,
                                (0, 0, 0), eraser_size * 2
                            )
                        else:
                            draw_segment(
                                canvas, previous, current,
                                COLORS[color_index][1], thickness
                            )

                    previous = current
                    drawing = True

                else:
                    gesture = "READY"
                    drawing = False
                    previous = None
                    smooth_point = None

                # Draw tracked fingertip marker on the camera image.
                fx = int(index_tip.x * CAM_W)
                fy = int(index_tip.y * CAM_H)
                cv2.circle(
                    frame, (fx, fy), 9, (0, 255, 0), -1
                )

            else:
                drawing = False
                previous = None
                smooth_point = None

            # Display board
            screen = canvas.copy()

            # Header background
            cv2.rectangle(
                screen, (0, 0), (W, 112), (25, 25, 25), -1
            )

            draw_text(
                screen, "FINGERWALL", (20, 34), 0.8,
                (255, 255, 255), 2
            )

            status_color = (
                (0, 255, 0) if gesture == "DRAWING"
                else (0, 200, 255)
            )
            draw_text(
                screen, gesture, (20, 66), 0.55,
                status_color, 2
            )

            # Colour palette
            for i, (name, color) in enumerate(COLORS):
                cx = 245 + i * 58
                cv2.circle(
                    screen, (cx, 36), 15, color, -1,
                    cv2.LINE_AA
                )

                if i == color_index:
                    cv2.circle(
                        screen, (cx, 36), 20,
                        (255, 255, 255), 2, cv2.LINE_AA
                    )

                draw_text(
                    screen, str(i + 1), (cx - 4, 68),
                    0.5, (230, 230, 230), 1
                )

            draw_text(
                screen, "Pen: " + str(thickness),
                (560, 68), 0.5
            )
            draw_text(
                screen,
                "ERASER ON" if eraser_mode else "PEN ON",
                (690, 68), 0.5,
                (0, 200, 255) if eraser_mode
                else (255, 255, 255), 1
            )

            # Small live camera preview
            if show_camera:
                preview = cv2.resize(frame, (192, 144))
                px, py = W - 207, 125
                screen[py:py + 144, px:px + 192] = preview
                cv2.rectangle(
                    screen, (px, py), (px + 192, py + 144),
                    (255, 255, 255), 2
                )
                draw_text(
                    screen, "LIVE CAMERA", (px, py - 8),
                    0.4, (255, 255, 255), 1
                )

            draw_text(
                screen,
                "1-5 Colour | +/- Size | E Eraser | V Camera",
                (20, H - 47), 0.48
            )
            draw_text(
                screen, "C Clear | S Save | Q Quit",
                (20, H - 20), 0.48
            )

            cv2.imshow(window, screen)
            key = cv2.waitKey(1) & 0xFF

            # Keyboard controls
            if key == ord("q"):
                break

            elif key == ord("c"):
                canvas[:] = 0
                previous = None
                smooth_point = None
                drawing = False
                print("Board cleared.")

            elif key == ord("s"):
                last_saved = save_board()

            elif key == ord("v"):
                show_camera = not show_camera

            elif key == ord("e"):
                eraser_mode = not eraser_mode
                previous = None
                smooth_point = None
                print(
                    "Eraser mode ON" if eraser_mode
                    else "Pen mode ON"
                )

            elif key in [ord("+"), ord("=")]:
                thickness = min(25, thickness + 2)
                print("Pen thickness:", thickness)

            elif key in [ord("-"), ord("_")]:
                thickness = max(1, thickness - 2)
                print("Pen thickness:", thickness)

            elif ord("1") <= key <= ord("5"):
                color_index = key - ord("1")
                eraser_mode = False
                print("Colour:", COLORS[color_index][0])

finally:
    camera.release()
    cv2.destroyAllWindows()
