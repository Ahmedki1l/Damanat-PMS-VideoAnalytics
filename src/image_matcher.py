"""
image_matcher.py — Multi-feature vehicle image matcher for ANPR re-identification.

Provides robust matching between ANPR entrance images and parking camera detection crops.
Uses multiple features to improve accuracy across different camera angles and lighting:

  1. Dominant Color Extraction (K-means in LAB color space)
     - Most viewpoint-invariant feature: a red car is red from any angle
     - Uses perceptually uniform LAB space for better color discrimination
     - Extracts top 3 dominant colors weighted by cluster size

  2. Regional Color Comparison
     - Splits car image into top/bottom halves (roof vs body)
     - Captures two-tone patterns (e.g., dark roof + white body)

  3. Structural Similarity Index (SSIM)
     - Measures luminance, contrast, and structure patterns
     - More robust than simple pixel comparison

  4. Edge Distribution
     - Canny edge density comparison
     - Cars with more/fewer visible features score differently
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


class VehicleImageMatcher:
    """
    Multi-feature vehicle image matching for cross-camera re-identification.

    Usage:
        matcher = VehicleImageMatcher()
        score = matcher.compare(anpr_image, camera_crop)
        # score: 0.0 (different) to 1.0 (identical)
    """

    # Feature weights for final score
    WEIGHT_DOMINANT_COLOR = 0.45   # Most reliable across viewpoints
    WEIGHT_REGIONAL_COLOR = 0.25  # Two-tone pattern matching
    WEIGHT_SSIM = 0.15            # Structural similarity
    WEIGHT_EDGE = 0.15            # Edge density comparison

    # Minimum crop size to process (pixels)
    MIN_CROP_SIZE = 20

    def __init__(self, n_colors: int = 3):
        """
        Args:
            n_colors: Number of dominant colors to extract (default 3).
        """
        self.n_colors = n_colors

    def compare(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compare two vehicle images using multiple features.

        Args:
            img1: First vehicle image (BGR, e.g., ANPR capture).
            img2: Second vehicle image (BGR, e.g., camera detection crop).

        Returns:
            Similarity score between 0.0 (different) and 1.0 (very similar).
        """
        if img1 is None or img2 is None:
            return 0.0
        if img1.size == 0 or img2.size == 0:
            return 0.0
        if (img1.shape[0] < self.MIN_CROP_SIZE or img1.shape[1] < self.MIN_CROP_SIZE or
                img2.shape[0] < self.MIN_CROP_SIZE or img2.shape[1] < self.MIN_CROP_SIZE):
            return 0.0

        try:
            scores = {}

            # --- 1. Dominant color matching ---
            scores["dominant_color"] = self._compare_dominant_colors(img1, img2)

            # --- 2. Regional color comparison ---
            scores["regional_color"] = self._compare_regional_colors(img1, img2)

            # --- 3. Structural similarity ---
            scores["ssim"] = self._compute_ssim(img1, img2)

            # --- 4. Edge density comparison ---
            scores["edge"] = self._compare_edges(img1, img2)

            # Weighted combination
            final = (
                scores["dominant_color"] * self.WEIGHT_DOMINANT_COLOR +
                scores["regional_color"] * self.WEIGHT_REGIONAL_COLOR +
                scores["ssim"] * self.WEIGHT_SSIM +
                scores["edge"] * self.WEIGHT_EDGE
            )

            return max(0.0, min(1.0, final))

        except Exception:
            return 0.0

    def _extract_dominant_colors(
        self, img: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Extract dominant colors using K-means clustering in LAB space.

        Returns list of (lab_color, weight) sorted by weight descending.
        Weight = proportion of pixels in that cluster.
        """
        # Resize for speed
        small = cv2.resize(img, (64, 64))

        # Convert to LAB (perceptually uniform — distances = perceived differences)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = min(self.n_colors, len(pixels))
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # Compute cluster weights (proportion of pixels)
        total = len(labels)
        colors = []
        for i in range(k):
            weight = np.sum(labels == i) / total
            colors.append((centers[i], weight))

        # Sort by weight (most dominant first)
        colors.sort(key=lambda x: x[1], reverse=True)
        return colors

    def _compare_dominant_colors(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compare dominant colors between two images.

        Uses LAB distance between dominant color clusters.
        Matches each color from img1 to the closest color in img2.
        """
        colors1 = self._extract_dominant_colors(img1)
        colors2 = self._extract_dominant_colors(img2)

        if not colors1 or not colors2:
            return 0.0

        # For each dominant color in img1, find best match in img2
        total_score = 0.0
        total_weight = 0.0

        for c1, w1 in colors1:
            best_dist = float("inf")
            for c2, w2 in colors2:
                # Euclidean distance in LAB space
                # In LAB, distance < 10 = barely noticeable, < 25 = similar shade
                dist = np.linalg.norm(c1 - c2)
                if dist < best_dist:
                    best_dist = dist

            # Convert distance to similarity [0, 1]
            # LAB distance ~0 = identical, ~50 = very different, ~100+ = opposite
            similarity = max(0.0, 1.0 - (best_dist / 60.0))
            total_score += similarity * w1
            total_weight += w1

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def _compare_regional_colors(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compare images by splitting into top/bottom halves.

        Captures two-tone patterns (roof color vs body color).
        """
        h1 = img1.shape[0] // 2
        h2 = img2.shape[0] // 2

        if h1 < self.MIN_CROP_SIZE or h2 < self.MIN_CROP_SIZE:
            return 0.0

        # Top halves (roof/upper body)
        top_score = self._compare_mean_color(img1[:h1], img2[:h2])
        # Bottom halves (lower body/wheels)
        bot_score = self._compare_mean_color(img1[h1:], img2[h2:])

        return (top_score + bot_score) / 2.0

    @staticmethod
    def _compare_mean_color(region1: np.ndarray, region2: np.ndarray) -> float:
        """Compare mean colors of two regions in LAB space."""
        lab1 = cv2.cvtColor(cv2.resize(region1, (32, 32)), cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(cv2.resize(region2, (32, 32)), cv2.COLOR_BGR2LAB)

        mean1 = np.mean(lab1.reshape(-1, 3), axis=0)
        mean2 = np.mean(lab2.reshape(-1, 3), axis=0)

        dist = np.linalg.norm(mean1 - mean2)
        return max(0.0, 1.0 - (dist / 60.0))

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute structural similarity between two images.

        Manual SSIM implementation (no scikit-image dependency).
        """
        size = (128, 128)
        g1 = cv2.cvtColor(cv2.resize(img1, size), cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(cv2.resize(img2, size), cv2.COLOR_BGR2GRAY).astype(np.float64)

        # SSIM constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # Mean SSIM, clamped to [0, 1]
        return max(0.0, float(np.mean(ssim_map)))

    @staticmethod
    def _compare_edges(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compare edge density patterns between two images.

        Uses Canny edge detection + regional density comparison.
        """
        size = (64, 64)
        g1 = cv2.cvtColor(cv2.resize(img1, size), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(img2, size), cv2.COLOR_BGR2GRAY)

        edges1 = cv2.Canny(g1, 50, 150)
        edges2 = cv2.Canny(g2, 50, 150)

        # Split into 4 quadrants and compare density per quadrant
        h, w = size[1], size[0]
        quadrant_scores = []

        for ry in range(2):
            for rx in range(2):
                y1, y2 = ry * h // 2, (ry + 1) * h // 2
                x1, x2 = rx * w // 2, (rx + 1) * w // 2

                d1 = np.mean(edges1[y1:y2, x1:x2]) / 255.0
                d2 = np.mean(edges2[y1:y2, x1:x2]) / 255.0

                # Similarity = 1.0 when densities are equal
                diff = abs(d1 - d2)
                quadrant_scores.append(max(0.0, 1.0 - diff * 3.0))

        return sum(quadrant_scores) / len(quadrant_scores)
