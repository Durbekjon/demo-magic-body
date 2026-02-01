"""
Magic Body - Fun-House Body Effects Demo
Lightweight version for desktop packaging with PyInstaller.
Uses geometric warping (TPS/RBF) for real-time effects.
"""
import cv2
import mediapipe as mp
import time
import numpy as np
import sys
import os

# --- Helper to find bundled resources ---
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# --- CONFIGURATION ---
ENABLE_POSE = True
ENABLE_SEGMENTATION = True
ENABLE_EFFECTS = True

SMOOTHING_ALPHA = 0.75
EFFECT_MODE = 1

EFFECT_PARAMS = {
    1: {"name": "Big Head", "head": 0.5, "arms": 0.0, "legs": 0.0, "torso": 0.0, "rubber": 0.0},
    2: {"name": "Long Limbs", "head": 0.0, "arms": 0.35, "legs": 0.4, "torso": 0.0, "rubber": 0.0},
    3: {"name": "Rubber Body", "head": 0.25, "arms": 0.2, "legs": 0.2, "torso": 0.15, "rubber": 0.5},
    4: {"name": "Alien", "head": 0.55, "arms": 0.3, "legs": 0.35, "torso": -0.25, "rubber": 0.0},
}

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

HEAD_LANDMARKS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "mouth_left", "mouth_right"]
LEFT_ARM = ["left_shoulder", "left_elbow", "left_wrist"]
RIGHT_ARM = ["right_shoulder", "right_elbow", "right_wrist"]
LEFT_LEG = ["left_hip", "left_knee", "left_ankle"]
RIGHT_LEG = ["right_hip", "right_knee", "right_ankle"]

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), 
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"), 
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"), 
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle")
]

class LandmarkSmoother:
    def __init__(self, alpha=0.75):
        self.alpha = alpha
        self.prev = {}

    def smooth(self, pts):
        out = {}
        for k, (x, y) in pts.items():
            if k in self.prev:
                px, py = self.prev[k]
                x = self.alpha * x + (1 - self.alpha) * px
                y = self.alpha * y + (1 - self.alpha) * py
            out[k] = (int(x), int(y))
        self.prev = out
        return out

class PoseDataLayer:
    @staticmethod
    def extract(result, w, h):
        if not result.pose_landmarks:
            return {}
        pts = {}
        for i, lm in enumerate(result.pose_landmarks[0]):
            if lm.visibility > 0.5 and i < len(LANDMARK_NAMES):
                pts[LANDMARK_NAMES[i]] = (int(lm.x * w), int(lm.y * h))
        return pts

class SegmentationLayer:
    def __init__(self):
        model_path = resource_path('selfie_segmenter.tflite')
        opts = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True
        )
        self.seg = ImageSegmenter.create_from_options(opts)

    def get_mask(self, img):
        res = self.seg.segment(img)
        m = res.category_mask.numpy_view()
        m = (m > 0.2).astype(np.uint8) * 255
        return cv2.GaussianBlur(m, (21, 21), 0)

    def close(self):
        self.seg.close()

class EffectsEngine:
    """Fast geometric effects using displacement fields."""
    
    def __init__(self):
        self.prev_dx = None
        self.prev_dy = None
        self.frame_t = 0
        
    def apply(self, frame, pts, mask, params):
        self.frame_t += 1
        if not pts or mask is None:
            return frame
        
        h, w = frame.shape[:2]
        orig = frame.copy()
        
        # Create base coordinate maps
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        map_x, map_y = np.meshgrid(x_coords, y_coords)
        
        y_grid, x_grid = np.ogrid[:h, :w]
        dx = np.zeros((h, w), dtype=np.float32)
        dy = np.zeros((h, w), dtype=np.float32)
        
        # Head effect
        head_str = params.get("head", 0)
        if head_str != 0 and "nose" in pts:
            cx, cy = pts["nose"]
            radius = 80
            if "left_ear" in pts and "right_ear" in pts:
                radius = int(abs(pts["left_ear"][0] - pts["right_ear"][0]) * 0.8)
            radius = max(50, min(150, radius))
            
            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2).astype(np.float32)
            influence = np.exp(-0.5 * (dist / radius)**2)
            strength = head_str * radius * 0.5
            
            dx += (cx - x_grid) * influence * strength / (dist + 1)
            dy += (cy - y_grid) * influence * strength / (dist + 1)
        
        # Limb stretching
        for chain, strength_key in [(LEFT_ARM, "arms"), (RIGHT_ARM, "arms"), 
                                     (LEFT_LEG, "legs"), (RIGHT_LEG, "legs")]:
            str_val = params.get(strength_key, 0)
            if str_val != 0 and all(n in pts for n in chain):
                self._apply_limb(dx, dy, pts, chain, str_val, w, h, x_grid, y_grid)
        
        # Rubber wobble
        if params.get("rubber", 0) > 0:
            t = self.frame_t * 0.12
            str_val = params["rubber"]
            wave_x = np.sin(t + y_grid * 0.04) * str_val * 12
            wave_y = np.sin(t * 1.1 + x_grid * 0.03) * str_val * 8
            dx += wave_x.astype(np.float32)
            dy += wave_y.astype(np.float32)
        
        # Temporal smoothing
        if self.prev_dx is not None:
            alpha = 0.6
            dx = alpha * dx + (1 - alpha) * self.prev_dx
            dy = alpha * dy + (1 - alpha) * self.prev_dy
        self.prev_dx, self.prev_dy = dx.copy(), dy.copy()
        
        # Apply within mask
        mask_norm = (mask / 255.0).astype(np.float32)
        mask_blur = cv2.GaussianBlur(mask_norm, (31, 31), 0)
        
        final_x = map_x - dx * mask_blur
        final_y = map_y - dy * mask_blur
        
        warped = cv2.remap(frame, final_x, final_y, cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REFLECT)
        
        mask_3ch = np.stack([mask_blur]*3, axis=-1)
        result = orig * (1 - mask_3ch) + warped * mask_3ch
        
        return result.astype(np.uint8)
    
    def _apply_limb(self, dx, dy, pts, chain, strength, w, h, x_grid, y_grid):
        start = np.array(pts[chain[0]], dtype=np.float32)
        end = np.array(pts[chain[-1]], dtype=np.float32)
        
        limb_vec = end - start
        limb_len = np.linalg.norm(limb_vec) + 1e-6
        limb_dir = limb_vec / limb_len
        
        px = x_grid - start[0]
        py = y_grid - start[1]
        proj = (px * limb_dir[0] + py * limb_dir[1]).astype(np.float32)
        perp_dist = np.abs(px * (-limb_dir[1]) + py * limb_dir[0])
        
        width = 40
        influence = np.exp(-0.5 * (perp_dist / width)**2)
        t = np.clip(proj / limb_len, 0, 1)
        stretch = t ** 2 * strength * limb_len * 0.4
        
        dx -= limb_dir[0] * stretch * influence
        dy -= limb_dir[1] * stretch * influence

def render_ui(frame, pts, fps, mode):
    if ENABLE_POSE and pts:
        for s, e in SKELETON_CONNECTIONS:
            if s in pts and e in pts:
                cv2.line(frame, pts[s], pts[e], (255, 255, 255), 2)
        for c in pts.values():
            cv2.circle(frame, c, 4, (0, 255, 0), -1)
    
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    name = EFFECT_PARAMS.get(mode, {}).get("name", "?")
    cv2.putText(frame, f"[{mode}] {name}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, "1-4: effects | ESC: exit", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

def main():
    global EFFECT_MODE
    
    print("=" * 40)
    print("  Magic Body - Fun-House Effects")
    print("=" * 40)
    
    smoother = LandmarkSmoother(SMOOTHING_ALPHA)
    engine = EffectsEngine()
    
    pose_model = resource_path('pose_landmarker.task')
    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5, min_tracking_confidence=0.5
    )

    try:
        landmarker = PoseLandmarker.create_from_options(pose_opts)
        segmenter = SegmentationLayer()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure pose_landmarker.task and selfie_segmenter.tflite are present.")
        input("Press Enter to exit...")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        input("Press Enter to exit...")
        return

    print("\nControls: 1-4 = effects, ESC = exit")
    print("Starting...")
    
    prev_t = 0
    
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            res = landmarker.detect(mp_img)
            raw_pts = PoseDataLayer.extract(res, w, h)
            pts = smoother.smooth(raw_pts) if raw_pts else {}

            mask = segmenter.get_mask(mp_img)

            if ENABLE_EFFECTS and mask is not None:
                params = EFFECT_PARAMS.get(EFFECT_MODE, EFFECT_PARAMS[1])
                frame = engine.apply(frame, pts, mask, params)

            curr_t = time.time()
            fps = 1 / (curr_t - prev_t) if curr_t != prev_t else 0
            prev_t = curr_t

            render_ui(frame, pts, fps, EFFECT_MODE)
            cv2.imshow('Magic Body', frame)
            
            key = cv2.waitKey(5) & 0xFF
            if key == 27: break
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                EFFECT_MODE = key - ord('0')
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        segmenter.close()
        print("Closed.")

if __name__ == "__main__":
    main()
