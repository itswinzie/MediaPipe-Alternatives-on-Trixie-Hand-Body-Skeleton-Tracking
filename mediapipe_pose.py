import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    model_complexity=1,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        out = frame.copy()

        # 1) BODY MASK (badan)
        if res.segmentation_mask is not None:
            mask = res.segmentation_mask
            mask = cv2.resize(mask, (w, h))
            mask_bin = (mask > 0.3).astype(np.uint8)  # threshold

            # buat “badan” versi fill (contoh: putih)
            body_layer = np.full_like(out, 255)  # putih
            out = out * (1 - mask_bin[..., None]) + body_layer * (mask_bin[..., None])
            out = out.astype(np.uint8)

        # 2) SKELETON (pose)
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                out,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

        cv2.imshow("Version A: Skeleton + Body", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
