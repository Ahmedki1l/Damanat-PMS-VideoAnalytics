"""
reid_matcher.py — Deep Learning ReID feature extraction for vehicles.

Uses the Omni-Scale Network (OSNet-AIN) to extract 512-D embeddings that are
robust to changes in lighting, perspective, and camera quality.
"""

import logging
import torch
import numpy as np
import cv2
from typing import Optional, List
from src.reid_matcher.reid_preprocessing import normalize_for_reid
from src.config import ReIDPreprocessingConfig

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

    def __init__(
        self,
        model_name: str = 'osnet_ain_x1_0',
        use_gpu: bool = True,
        batch_size: int = 16,
        preprocessing_config: ReIDPreprocessingConfig = None,
    ):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.batch_size = batch_size
        self.preprocessing_config = preprocessing_config or ReIDPreprocessingConfig()
        
        logger.info(f"[REID] Initializing {model_name} on {self.device}...")
        pp_status = "ON" if self.preprocessing_config.enabled else "OFF"
        logger.info(f"[REID] Preprocessing: {pp_status}")
        
        if build_model is None:
            raise ImportError("torchreid not found. Please install with 'pip install torchreid'")

        # Build model
        self.model = build_model(
            name=model_name,
            num_classes=1000, # Match ImageNet weights to avoid warnings; eval() returns features.
            pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing parameters
        self.input_size = (128, 256)  # (Height, Width) - better for vehicles
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Letterbox resize and normalize a single BGR image.
        
        Args:
            image: BGR image from cv2.
            
        Returns:
            Preprocessed tensor (3, H, W).
        """
        # Apply luminance normalization if enabled
        pp = self.preprocessing_config
        if pp.enabled:
            image = normalize_for_reid(
                image,
                clip_limit=pp.clip_limit,
                grid_size=pp.grid_size,
            )

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Letterbox resize (preserve aspect ratio with padding)
        h, w = image.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas and paste resized image
        canvas = np.full((target_h, target_w, 3), 127, dtype=np.uint8) # Grey padding
        dw, dh = (target_w - nw) // 2, (target_h - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw, :] = image_resized
        
        # To Tensor and Normalize
        img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.norm_mean).view(3, 1, 1)
        std = torch.tensor(self.norm_std).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor

    @torch.no_grad()
    def extract_feature(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512-D feature vector from a single BGR image crop.
        """
        if image is None or image.size == 0:
            return None
        
        try:
            input_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
            features = self.model(input_tensor)
            
            vec = features.cpu().numpy()[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception as e:
            logger.error(f"[REID] Single extraction error: {e}")
            return None

    @torch.no_grad()
    def extract_features_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract features for a list of images in batches.
        
        Returns:
            List of normalized 512-D vectors (or None for failed extractions).
        """
        if not images:
            return []

        results = [None] * len(images)
        valid_indices = [i for i, img in enumerate(images) if img is not None and img.size > 0]
        
        if not valid_indices:
            return results

        valid_images = [images[i] for i in valid_indices]
        
        for i in range(0, len(valid_images), self.batch_size):
            batch_images = valid_images[i : i + self.batch_size]
            batch_indices = valid_indices[i : i + self.batch_size]
            
            try:
                tensors = [self._preprocess(img) for img in batch_images]
                input_batch = torch.stack(tensors).to(self.device)
                
                features = self.model(input_batch)
                features_np = features.cpu().numpy()
                
                for idx_in_batch, global_idx in enumerate(batch_indices):
                    vec = features_np[idx_in_batch]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                    results[global_idx] = vec
                    
            except Exception as e:
                logger.error(f"[REID] Batch extraction error at indices {batch_indices}: {e}")

        return results

    @staticmethod
    def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized vectors.
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        return float(np.dot(feat1, feat2))

# Thread-safe singleton
_matcher_instance = None
_matcher_lock = __import__("threading").Lock()

def get_reid_matcher(preprocessing_config: ReIDPreprocessingConfig = None) -> VehicleReIDMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = VehicleReIDMatcher(
                    preprocessing_config=preprocessing_config,
                )
    return _matcher_instance
