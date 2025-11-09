# SPDX-License-Identifier: Apache-2.0
"""
SVD-XKV Deserializer:
Reconstructs full-precision KV tensors from cross-layer SVD representation.
Implements the deserialization half of remote_serde: "svd_xkv".
"""

# Standard
import io
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.serde import Deserializer
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _torch_load_from_bytes(payload: bytes) -> dict:
    """Load serialized tensors from torch.save() payload."""
    buf = io.BytesIO(payload)
    return torch.load(buf, map_location="cpu")

class SvdXKVDeserializer(Deserializer):
    """
    Decode the compressed SVD payload (A_k, B_k, A_v, B_v)
    back into the full KV tensor expected by vLLM / LMCache.
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata, dtype):
        self.dtype = dtype
        self.fmt = metadata.fmt  # "vllm" or "huggingface"
        self.chunk_size = config.chunk_size

    @_lmcache_nvtx_annotate
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        """
        Decode the SVD-compressed representation back into
        [num_layers, 2, num_tokens, num_heads, head_size] tensor.
        """
        t0 = time.time()

        # Load serialized dict of tensors
        obj = _torch_load_from_bytes(bs)
        meta: torch.Tensor = obj["meta"].to(torch.int32)
        L, T, H, D, rk, rv = meta.tolist()
        hidden = H * D

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        A_k = obj["A_k"].to(device=device, dtype=self.dtype)  # [hidden, rk]
        B_k = obj["B_k"].to(device=device, dtype=self.dtype)  # [L*T, rk]
        A_v = obj["A_v"].to(device=device, dtype=self.dtype)  # [hidden, rv]
        B_v = obj["B_v"].to(device=device, dtype=self.dtype)  # [L*T, rv]

        # Reconstruct K and V
        K = (B_k @ A_k.T).reshape(L, T, hidden)
        V = (B_v @ A_v.T).reshape(L, T, hidden)

        # Restore heads
        K = K.reshape(L, T, H, D)
        V = V.reshape(L, T, H, D)

        # Merge into [L, 2, T, H, D]
        kv = torch.stack([K, V], dim=1).to(self.dtype)

        # Handle tensor format
        match self.fmt:
            case "vllm":
                # [L, 2, T, H, D]
                pass
            case "huggingface":
                # Convert back to [L, 2, H, T, D] (HF expects heads before seq)
                kv = kv.permute(0, 1, 3, 2, 4)
            case _:
                raise RuntimeError(f"Unknown format {self.fmt}")

        t1 = time.time()
        logger.info(
            f"[SVD-XKV] Decoded: L={L}, T={T}, H={H}, D={D}, "
            f"rk={rk}, rv={rv}, time={t1 - t0:.3f}s"
        )

        return kv