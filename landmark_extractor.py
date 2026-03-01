"""
Vocalize — Phase 1: Real-time MediaPipe Holistic Landmark Extraction

Opens the webcam, applies MediaPipe Holistic (face, pose, both hands),
draws landmarks on screen, and extracts all keypoints into a flattened
NumPy array of shape (1662,).

Controls:
    s  — save current frame's keypoints to  output/keypoints.npy
    q  — quit
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# ──────────────────────────────────────────────
# MediaPipe shortcuts
# ──────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────

def mediapipe_detect(image: np.ndarray, model: mp_holistic.Holistic):
    """
    Convert a BGR frame to RGB, run MediaPipe Holistic detection,
    then convert back to BGR so the caller can keep using OpenCV.

    Args:
        image:  BGR frame from cv2.VideoCapture.
        model:  An initialised mp.solutions.holistic.Holistic instance.

    Returns:
        image:   The same frame (BGR, writeable).
        results: MediaPipe Holistic results object.
    """
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    image.flags.writeable = True
    return image, results


def draw_landmarks(image: np.ndarray, results) -> np.ndarray:
    """
    Draw face, pose, left-hand, and right-hand landmarks/connections
    onto the frame using MediaPipe drawing utilities.

    Args:
        image:   BGR frame.
        results: MediaPipe Holistic results.

    Returns:
        The annotated frame (same object, modified in-place).
    """
    # Face mesh tesselation
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )

    # Pose skeleton
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )

    # Left hand
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )

    # Right hand
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

    return image


def extract_keypoints(results) -> np.ndarray:
    """
    Flatten all detected landmarks into a single 1-D NumPy array
    using **nose-relative** coordinates for translation invariance.

    The nose (Pose Landmark 0) is used as the origin.  Its (x, y, z)
    is subtracted from every other landmark so the data is
    independent of where the user sits in the frame.

    Layout (1662 floats total):
        pose       — 33 landmarks × 4 (x, y, z, visibility) = 132
        face       — 468 landmarks × 3 (x, y, z)            = 1404
        left_hand  — 21 landmarks × 3 (x, y, z)             = 63
        right_hand — 21 landmarks × 3 (x, y, z)             = 63

    Missing landmark groups are zero-padded so the output shape is
    always (1662,) regardless of detection quality.

    Args:
        results: MediaPipe Holistic results.

    Returns:
        np.ndarray of shape (1662,) and dtype float64.
    """
    # ── Reference point: nose (Pose Landmark 0) ──
    nose_x, nose_y, nose_z = 0.0, 0.0, 0.0
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        nose_x, nose_y, nose_z = nose.x, nose.y, nose.z

    # ── Pose: (x-nose_x, y-nose_y, z-nose_z, visibility) ──
    if results.pose_landmarks:
        pose = np.array([
            [lm.x - nose_x, lm.y - nose_y, lm.z - nose_z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # ── Face: (x-nose_x, y-nose_y, z-nose_z) ──
    if results.face_landmarks:
        face = np.array([
            [lm.x - nose_x, lm.y - nose_y, lm.z - nose_z]
            for lm in results.face_landmarks.landmark
        ]).flatten()
    else:
        face = np.zeros(468 * 3)

    # ── Left hand ──
    if results.left_hand_landmarks:
        left_hand = np.array([
            [lm.x - nose_x, lm.y - nose_y, lm.z - nose_z]
            for lm in results.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left_hand = np.zeros(21 * 3)

    # ── Right hand ──
    if results.right_hand_landmarks:
        right_hand = np.array([
            [lm.x - nose_x, lm.y - nose_y, lm.z - nose_z]
            for lm in results.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right_hand = np.zeros(21 * 3)

    return np.concatenate([pose, face, left_hand, right_hand])


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def main():
    """
    Open the default webcam, run MediaPipe Holistic on each frame,
    display annotated video, and allow the user to save keypoints.
    """
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    print("=" * 50)
    print("  Vocalize — Phase 1  Landmark Extractor")
    print("=" * 50)
    print("  Press 's' to save keypoints  |  Press 'q' to quit")
    print("=" * 50)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("[WARN] Ignoring empty frame.")
                continue

            # ---- detect & draw ----
            frame, results = mediapipe_detect(frame, holistic)
            draw_landmarks(frame, results)

            # ---- show FPS on frame ----
            cv2.putText(
                frame,
                "Press 's' save | 'q' quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Vocalize — Landmark Extractor", frame)

            # ---- handle key presses ----
            key = cv2.waitKey(5) & 0xFF

            if key == ord("q"):
                print("\n[INFO] Quitting...")
                break

            if key == ord("s"):
                keypoints = extract_keypoints(results)
                save_path = os.path.join(output_dir, "keypoints.npy")
                np.save(save_path, keypoints)
                print(f"[SAVED] Keypoints shape: {keypoints.shape}  →  {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    main()
