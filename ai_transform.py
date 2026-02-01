"""
AI Transformation Pipeline using Stable Diffusion
Provides intelligent, neural network-based body deformation.
"""
import torch
from PIL import Image
import numpy as np

class AITransformer:
    """Wraps Stable Diffusion img2img for body transformation."""
    
    # Effect prompts for different transformation modes
    PROMPTS = {
        1: {
            "name": "Big Head Cartoon",
            "prompt": "cartoon character with oversized head, big eyes, chibi style, exaggerated proportions, same pose, same clothing",
            "negative": "realistic, normal proportions, photograph"
        },
        2: {
            "name": "Long Limbs Alien", 
            "prompt": "alien creature with elongated limbs, stretched arms and legs, surreal proportions, same pose",
            "negative": "normal human, realistic proportions"
        },
        3: {
            "name": "Rubber Body",
            "prompt": "rubber toy figure, bendy limbs, plastic skin, toy-like proportions, same pose",
            "negative": "realistic, human skin, normal anatomy"
        },
        4: {
            "name": "Funhouse Mirror",
            "prompt": "funhouse mirror distortion, wavy body, surreal warped proportions, dreamlike, same person",
            "negative": "normal, undistorted, realistic"
        }
    }
    
    def __init__(self, model_id="stabilityai/sdxl-turbo", device=None):
        """Initialize the diffusion pipeline."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.model_id = model_id
        self.ready = False
        
    def load(self):
        """Load the model (call once at startup)."""
        try:
            from diffusers import AutoPipelineForImage2Image
            
            print(f"Loading {self.model_id} on {self.device}...")
            
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None,
            )
            self.pipe.to(self.device)
            
            # Optimize for speed
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass  # xformers not available
            
            self.ready = True
            print("AI Transformer ready!")
            return True
            
        except Exception as e:
            print(f"Failed to load AI model: {e}")
            if "MT5Tokenizer" in str(e) or "sentencepiece" in str(e):
                print("MISSING DEPENDENCY: Please run: pip install sentencepiece protobuf")
            print("Install dependencies: pip install diffusers transformers accelerate torch sentencepiece")
            return False
    
    def transform(self, image, mode=1, strength=0.6, steps=4):
        """
        Transform an image using the selected effect mode.
        
        Args:
            image: numpy array (BGR) or PIL Image
            mode: effect mode 1-4
            strength: denoising strength (0.3-0.8 recommended)
            steps: inference steps (fewer = faster)
        
        Returns:
            Transformed PIL Image
        """
        if not self.ready:
            return image
        
        # Convert to PIL
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
            else:
                image = Image.fromarray(image)
        
        # Resize for speed
        orig_size = image.size
        target_size = (512, 512)
        image = image.resize(target_size, Image.LANCZOS)
        
        # Get prompt
        effect = self.PROMPTS.get(mode, self.PROMPTS[1])
        
        # Run inference
        with torch.inference_mode():
            result = self.pipe(
                prompt=effect["prompt"],
                negative_prompt=effect["negative"],
                image=image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=0.0,  # SDXL Turbo doesn't need guidance
            ).images[0]
        
        # Resize back
        result = result.resize(orig_size, Image.LANCZOS)
        
        return result
    
    def transform_to_numpy(self, image, mode=1, strength=0.6, steps=4):
        """Transform and return as BGR numpy array."""
        result = self.transform(image, mode, strength, steps)
        if isinstance(result, Image.Image):
            return np.array(result)[:, :, ::-1]  # RGB to BGR
        return result


class SegmentedTransformer:
    """Combines segmentation with AI transformation."""
    
    def __init__(self, ai_transformer):
        self.ai = ai_transformer
        self.last_transform = None
        self.frame_count = 0
        self.transform_interval = 5  # Transform every N frames
        
    def process(self, frame, mask, mode=1, strength=0.5):
        """
        Process a frame: transform person, keep background.
        
        Args:
            frame: BGR numpy array
            mask: segmentation mask (255=person)
            mode: effect mode
            strength: transformation strength
        
        Returns:
            Composited frame
        """
        self.frame_count += 1
        
        if not self.ai.ready:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create person-only image with white background
        mask_3ch = np.stack([mask/255]*3, axis=-1).astype(np.float32)
        person = (frame * mask_3ch).astype(np.uint8)
        
        # Add white background for clearer input
        white_bg = np.ones_like(frame) * 255
        person_on_white = (person * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
        
        # Transform (with frame skipping for performance)
        if self.frame_count % self.transform_interval == 0 or self.last_transform is None:
            try:
                transformed = self.ai.transform_to_numpy(person_on_white, mode, strength)
                self.last_transform = transformed
            except Exception as e:
                print(f"Transform error: {e}")
                self.last_transform = person_on_white
        
        transformed = self.last_transform
        
        # Resize if needed
        if transformed.shape[:2] != (h, w):
            transformed = cv2.resize(transformed, (w, h))
        
        # Composite: transformed person on original background
        # Expand mask slightly for smooth blending
        mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
        mask_norm = (mask_blur / 255.0).astype(np.float32)
        mask_3ch = np.stack([mask_norm]*3, axis=-1)
        
        result = (frame * (1 - mask_3ch) + transformed * mask_3ch).astype(np.uint8)
        
        return result


# For standalone testing
if __name__ == "__main__":
    import cv2
    
    print("Testing AI Transformer...")
    transformer = AITransformer()
    
    if transformer.load():
        # Test with a sample image
        test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = transformer.transform_to_numpy(test_img, mode=1)
        print(f"Transform successful! Output shape: {result.shape}")
    else:
        print("Could not load model. Check GPU and dependencies.")
