"""
Inference accelerators: fused LLR, encoder projection, optional row softmax (CuPy when available).
"""

from .cross_gemm import cross_r_dT
from .cuda_util import cuda_gemm_usable, warmup_cuda
from .gpu_stress import cupy_gemm_burn
from .score_batch import (
    fused_batch_infer_indices_confs_cupy,
    fused_features_to_device_latent_llr,
    fused_features_to_latent_and_llr,
    fused_score_llr,
    project_features,
    softmax_rows_llr,
)

__all__ = [
    "cross_r_dT",
    "cuda_gemm_usable",
    "cupy_gemm_burn",
    "fused_batch_infer_indices_confs_cupy",
    "fused_features_to_device_latent_llr",
    "fused_features_to_latent_and_llr",
    "fused_score_llr",
    "project_features",
    "softmax_rows_llr",
    "warmup_cuda",
]
