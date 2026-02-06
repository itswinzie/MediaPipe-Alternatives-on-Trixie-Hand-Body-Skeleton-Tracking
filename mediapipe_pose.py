import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "movenet_lightning.tflite"  # adjust if needed
CONF_TH = 0.30

# MoveNet keypoints order (17):
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
POSE_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,11),(6,12),(11,12),
    (5,7),(7,9),
    (6,8),(8,10),
    (11,13),(13,15),
    (12,14),(14,16)
]

def load_interpreter(model_path: str):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    return interpreter, in_det, out_det

def preprocess(frame_bgr, target_hw, input_dtype):
    h, w = target_hw
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h))

    if input_dtype == np.uint8:
        x = resized.astype(np.uint8)
    else:
        # float32 0..1
        x = (resized.astype(np.float32) / 255.0)

    return np.expand_dims(x, axis=0)

def main():
    interpreter, in_det, out_det = load_interpreter(MODEL_PATH)

    in_shape = in_det[0]["shape"]
    in_h, in_w = int(in_shape[1]), int(in_shape[2])
    in_dtype = in_det[0]["dtype"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try VideoCapture(0/1) or check permissions.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        inp = preprocess(frame, (in_h, in_w), in_dtype)
        interpreter.set_tensor(in_det[0]["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det[0]["index"])

        # Common MoveNet output: [1,1,17,3] with [y,x,score]
        kps = out[0, 0]  # [17,3]

        H, W = frame.shape[0], frame.shape[1]
        pts = []
        for i in range(17):
            y, x, s = kps[i]
            px, py = int(x * W), int(y * H)
            pts.append((px, py, float(s)))

        # draw edges
        for a, b in POSE_EDGES:
            if pts[a][2] > CONF_TH and pts[b][2] > CONF_TH:
                cv2.line(frame, (pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]), (255, 255, 255), 2)

        # draw points
        for x, y, s in pts:
            if s > CONF_TH:
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        cv2.imshow("MoveNet Pose Skeleton (ESC to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
