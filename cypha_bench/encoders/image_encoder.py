from __future__ import annotations

import numpy as np


class ImageEncoder:
    """Image feature encoders for greyscale 28x28 inputs."""

    def raw_pixels(self, img: np.ndarray) -> np.ndarray:
        return img.flatten().astype(np.float32) / 255.0

    def hog_features(self, img: np.ndarray, cell_size: int = 4, n_bins: int = 9) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) % np.pi

        n_cells_x = img.shape[1] // cell_size
        n_cells_y = img.shape[0] // cell_size
        hog = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)

        for b in range(n_bins):
            angle_low = b * np.pi / n_bins
            angle_high = (b + 1) * np.pi / n_bins
            mask = (angle >= angle_low) & (angle < angle_high)
            weighted = magnitude * mask
            for cy in range(n_cells_y):
                for cx in range(n_cells_x):
                    cell = weighted[
                        cy * cell_size : (cy + 1) * cell_size,
                        cx * cell_size : (cx + 1) * cell_size,
                    ]
                    hog[cy, cx, b] = cell.sum()
        return hog.flatten()

    def patch_features(self, img: np.ndarray, patch_size: int = 7, stride: int = 7) -> list[np.ndarray]:
        img_norm = img.astype(np.float32) / 255.0
        h, w = img.shape
        patches = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img_norm[y : y + patch_size, x : x + patch_size]
                patches.append(patch.flatten())
        return patches
