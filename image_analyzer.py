# Image Analyzer using OpenCLIP
# Extracts visual features from medical images for similarity search

import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Lazy loading to avoid import errors if not installed
def get_clip_model():
    """Load OpenCLIP model on first use (caches after)."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k'
    )
    model.eval()
    return model, preprocess

class ImageAnalyzer:
    """Analyzes medical images using OpenCLIP for embeddings."""
    
    def __init__(self):
        self.model = None
        self.preprocess = None
        self._loaded = False
    
    def _ensure_loaded(self):
        if not self._loaded:
            print("🖼️ Loading OpenCLIP model (first time only)...")
            self.model, self.preprocess = get_clip_model()
            self._loaded = True
            print("✅ OpenCLIP loaded successfully")
    
    def get_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        Extract 512-dim visual embedding from image bytes.
        
        Args:
            image_bytes: Raw image bytes (from file upload)
        
        Returns:
            512-dimensional numpy array
        """
        import torch
        self._ensure_loaded()
        
        # Load image
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess for CLIP
        image_tensor = self.preprocess(image).unsqueeze(0)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        
        return embedding.cpu().numpy().flatten()
    
    def image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string for API calls."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def describe_image_type(self, image_bytes: bytes) -> str:
        """
        Use CLIP zero-shot to classify image type.
        
        Returns one of: "x-ray", "ultrasound", "ct-scan", "lab-report", "other"
        """
        import torch
        import open_clip
        self._ensure_loaded()
        
        labels = [
            "a kidney x-ray image",
            "a kidney ultrasound image", 
            "a CT scan of kidneys",
            "a lab report document with blood test results",
            "a medical image"
        ]
        
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_tokens = tokenizer(labels)
        
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            best_idx = similarity.argmax().item()
        
        type_map = {0: "x-ray", 1: "ultrasound", 2: "ct-scan", 3: "lab-report", 4: "other"}
        return type_map[best_idx]


# Quick test
if __name__ == "__main__":
    analyzer = ImageAnalyzer()
    print("ImageAnalyzer ready for use.")
