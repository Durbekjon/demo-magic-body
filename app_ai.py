"""
AI-Powered Body Transformation Demo (AI Version)
Uses Stable Diffusion for intelligent, neural network-based deformation.
"""
import sys
import os

# Add local libs to path
# sys.path.append(os.path.join(os.path.dirname(__file__), "libs_ai"))

import cv2
import mediapipe as mp
import time
import numpy as np
import threading
from queue import Queue

# --- CONFIGURATION ---
ENABLE_SEGMENTATION = True
ENABLE_AI_TRANSFORM = True
EFFECT_MODE = 1  # 1-4 effect modes
TRANSFORM_STRENGTH = 0.5  # How much to transform (0.3-0.7)

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

EFFECT_NAMES = {
    1: "Big Head Cartoon",
    2: "Long Limbs Alien",
    3: "Rubber Toy",
    4: "Funhouse Mirror"
}

def resource_path(relative_path):
    """Get absolute path to resource."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

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


class AsyncAITransformer:
    """Asynchronous AI transformation for smooth demo experience."""
    
    def __init__(self):
        self.ai = None
        self.ready = False
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.last_result = None
        self.thread = None
        self.running = False
        self.error = None
        
    def load(self):
        """Load the AI model."""
        try:
            # Lazy import to avoid startup lag
            from ai_transform import AITransformer
            self.ai = AITransformer()
            if self.ai.load():
                self.ready = True
                self.running = True
                self.thread = threading.Thread(target=self._worker, daemon=True)
                self.thread.start()
                return True
        except ImportError as e:
            self.error = str(e)
            print(f"AI Transform module not available: {e}")
        except Exception as e:
            self.error = str(e)
            print(f"Failed to load AI: {e}")
        return False
    
    def _worker(self):
        """Background worker for AI transformation."""
        while self.running:
            try:
                if not self.input_queue.empty():
                    data = self.input_queue.get()
                    if data is None:
                        break
                    frame, mask, mode, strength = data
                    result = self._transform(frame, mask, mode, strength)
                    
                    # Clear old result and add new
                    while not self.output_queue.empty():
                        self.output_queue.get()
                    self.output_queue.put(result)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Worker error: {e}")
    
    def _transform(self, frame, mask, mode, strength):
        """Perform the actual transformation."""
        if not self.ready or self.ai is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create person on white background
        mask_norm = mask / 255.0
        mask_3ch = np.stack([mask_norm]*3, axis=-1).astype(np.float32)
        
        white_bg = np.ones_like(frame) * 255
        person_on_white = (frame * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
        
        # Transform
        try:
            transformed = self.ai.transform_to_numpy(person_on_white, mode, strength, steps=4)
            
            # Resize if needed
            if transformed.shape[:2] != (h, w):
                transformed = cv2.resize(transformed, (w, h))
            
            # Composite
            mask_blur = cv2.GaussianBlur(mask, (31, 31), 0)
            mask_blend = (mask_blur / 255.0).astype(np.float32)
            mask_blend_3ch = np.stack([mask_blend]*3, axis=-1)
            
            result = (frame * (1 - mask_blend_3ch) + transformed * mask_blend_3ch).astype(np.uint8)
            return result
            
        except Exception as e:
            print(f"Transform error: {e}")
            return frame
    
    def submit(self, frame, mask, mode, strength):
        """Submit a frame for transformation."""
        if not self.ready:
            return
        
        # Clear queue and add new frame
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                pass
        
        try:
            self.input_queue.put_nowait((frame.copy(), mask.copy(), mode, strength))
        except:
            pass
    
    def get_result(self, fallback):
        """Get latest transformed result or fallback."""
        if not self.output_queue.empty():
            self.last_result = self.output_queue.get()
        return self.last_result if self.last_result is not None else fallback
    
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.input_queue.put(None)
        if self.thread:
            self.thread.join(timeout=1)


def add_outline(frame, mask, thickness=2, color=(255, 255, 255)):
    """Add white outline around the person."""
    edges = cv2.Canny(mask, 50, 150)
    kernel = np.ones((thickness, thickness), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    frame[edges > 0] = color
    return frame


def render_ui(frame, fps, mode, ai_state):
    """Draw HUD."""
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    name = EFFECT_NAMES.get(mode, "Unknown")
    
    status_color = (0, 255, 0)
    if ai_state == "loading":
        status_text = "AI Loading..."
        status_color = (0, 165, 255) # Orange
    elif ai_state == "ready":
        status_text = "AI Ready"
        status_color = (0, 255, 0)   # Green
    else:
        status_text = "AI Failed"
        status_color = (0, 0, 255)   # Red
        
    cv2.putText(frame, f"[{mode}] {name}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, status_text, (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
    
    if ai_state == "failed":
         cv2.putText(frame, "Check console/libs", (20, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    else:
        cv2.putText(frame, "1-4:mode +/-:str ESC:exit", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)


def main():
    global EFFECT_MODE, TRANSFORM_STRENGTH
    
    print("=" * 50)
    print("AI Body Transformation (Pro Version)")
    print("=" * 50)
    
    # Initialize segmenter
    try:
        segmenter = SegmentationLayer()
    except Exception as e:
        print(f"Segmenter error: {e}")
        return
    
    # Initialize AI transformer (async)
    ai = AsyncAITransformer()
    ai_loading = threading.Thread(target=ai.load, daemon=True)
    ai_loading.start()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return
    
    prev_t = time.time()
    frame_count = 0
    
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_count += 1
            
            # Segment
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mask = segmenter.get_mask(mp_img)
            
            # Submit for AI transformation
            if ENABLE_AI_TRANSFORM and frame_count % 3 == 0:
                ai.submit(frame, mask, EFFECT_MODE, TRANSFORM_STRENGTH)
            
            # Get latest AI result or show segmented preview
            display = frame.copy()
            if ENABLE_AI_TRANSFORM and ai.ready:
                display = ai.get_result(frame)
            else:
                # Fallback
                if ENABLE_SEGMENTATION:
                    display = add_outline(display, mask)
            
            # FPS
            curr_t = time.time()
            fps = 1 / (curr_t - prev_t) if curr_t != prev_t else 0
            prev_t = curr_t
            
            ai_state = "ready" if ai.ready else "failed" if ai.error else "loading"
            render_ui(display, fps, EFFECT_MODE, ai_state)
            cv2.imshow('AI Body Transform', display)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                EFFECT_MODE = key - ord('0')
            elif key == ord('+') or key == ord('='):
                TRANSFORM_STRENGTH = min(0.8, TRANSFORM_STRENGTH + 0.1)
            elif key == ord('-'):
                TRANSFORM_STRENGTH = max(0.2, TRANSFORM_STRENGTH - 0.1)
                
    except KeyboardInterrupt:
        pass
    finally:
        ai.stop()
        cap.release()
        cv2.destroyAllWindows()
        segmenter.close()
        print("Done.")

if __name__ == "__main__":
    main()
