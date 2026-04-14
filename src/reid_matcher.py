"""
reid_matcher.py — Deep Learning ReID feature extraction for vehicles.

Uses the Omni-Scale Network (OSNet-AIN) to extract 512-D embeddings that are
robust to changes in lighting, perspective, and camera quality.
"""

import logging
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torchreid
    build_model = torchreid.models.build_model
    # No direct load_pretrained_weights in this version's init?
    # We'll use build_model(..., pretrained=True) anyway.
except ImportError as e:
    logger.warning(f"[REID] Failed to import torchreid: {e}")
    build_model = None
except Exception as e:
    logger.exception(f"[REID] Unexpected error during torchreid import: {e}")
    build_model = None

class VehicleReIDMatcher:
    """
    Extracts deep features from vehicle crops using OSNet.
    
    Provides 512-D normalized embeddings.
    Similarity is computed using Cosine Similarity (dot product of normalized vectors).
    """

    def __init__(self, model_name: str = 'osnet_ain_x1_0', use_gpu: bool = True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        logger.info(f"[REID] Initializing {model_name} on {self.device}...")
        
        if build_model is None:
            raise ImportError("torchreid not found. Please install with 'pip install torchreid'")

        # Build model
        self.model = build_model(
            name=model_name,
            num_classes=1, # Not training, so dummy value
            pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing pipeline
        # OSNet standard: 256x128, ImageNet normalization
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_feature(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512-D feature vector from a BGR image crop.
        
        Returns:
            unit-normalized numpy array (512,) or None if extraction fails.
        """
        if image is None or image.size == 0:
            return None

        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Transform and add batch dimension
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Extract features (output is [batch, 512])
            features = self.model(input_tensor)
            
            # Convert to numpy and normalize
            vec = features.cpu().numpy()[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                
            return vec
        except Exception as e:
            logger.error(f"[REID] Feature extraction error: {e}")
            return None

    @staticmethod
    def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized vectors.
        Output range: [-1, 1], though usually [0, 1] for these models.
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        # Dot product of normalized vectors = Cosine Similarity
        similarity = np.dot(feat1, feat2)
        return float(similarity)

# Thread-safe singleton
_matcher_instance = None
_matcher_lock = __import__("threading").Lock()

def get_reid_matcher() -> VehicleReIDMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = VehicleReIDMatcher()
    return _matcher_instance