import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "hand_landmark.tflite"  # adjust
CONF_TH = 0.30
MIRROR = True

# MediaPipe hand keypoints (21)
# 0 wrist
# 1-4 thumb (CMC, MCP, IP, tip)
# 5-8 index
# 9-12 middle
# 13-16 ring
# 17-20 pinky
HAND_EDGES = [
    # palm
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    # across knuckles (optional nicer skeleton)
    (5,9),(9,13),(13,17)
]

PALM_POLY = [0, 5, 9, 13, 17]  # wrist + MCPs

def load_interpreter(model_path: str):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    return interpreter, in_det, out_det

def preprocess(frame_bgr, target_hw, input_dtype):
    in_h, in_w = target_hw
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)

    if input_dtype == np.uint8:
        x = resized.astype(np.uint8)
    else:
        x = (resized.astype(np.float32) / 255.0)

    return np.expand_dims(x, axis=0)

def extract_hand_landmarks(output_tensor):
    """
    This depends on model.
    Expected common formats:
    - [1, 21, 3]  -> (x, y, score) OR (x, y, z)
    - [1, 1, 21, 3]
    We'll normalize to [21,3].
    """
    out = output_tensor
    if out.ndim == 4:
        # [1,1,21,3]
        return out[0, 0]
    elif out.ndim == 3:
        # [1,21,3]
        return out[0]
    else:
        raise ValueError(f"Unexpected output shape: {out.shape}")

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

        if MIRROR:
            frame = cv2.flip(frame, 1)

        inp = preprocess(frame, (in_h, in_w), in_dtype)
        interpreter.set_tensor(in_det[0]["index"], inp)
        interpreter.invoke()

        output = interpreter.get_tensor(out_det[0]["index"])
        lms = extract_hand_landmarks(output)  # [21,3]

        H, W = frame.shape[:2]

        # Many hand models output normalized x,y in [0..1]
        # Score may not exist. We'll treat 3rd value as score if it looks like [0..1].
        pts = []
        for i in range(21):
            x, y, s = lms[i]  # assume (x,y,score) normalized
            px, py = int(x * W), int(y * H)
            pts.append((px, py, float(s)))

        # draw edges
        for a, b in HAND_EDGES:
            if pts[a][2] > CONF_TH and pts[b][2] > CONF_TH:
                cv2.line(frame, (pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]), (255, 255, 255), 2)

        # draw points
        for x, y, s in pts:
            if s > CONF_TH:
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        # optional: fill palm (tapak tangan)
        poly = []
        ok_poly = True
        for idx in PALM_POLY:
            if pts[idx][2] <= CONF_TH:
                ok_poly = False
                break
            poly.append([pts[idx][0], pts[idx][1]])

        if ok_poly:
            poly = np.array([poly], dtype=np.int32)
            cv2.fillPoly(frame, poly, (255, 255, 255))

        cv2.imshow("TFLite Hand Skeleton (ESC to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
