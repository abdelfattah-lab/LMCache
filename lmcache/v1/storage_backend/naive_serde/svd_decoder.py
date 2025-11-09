# SPDX-License-Identifier: Apache-2.0
"""
SVD-XKV Decoder (v1-style)
Implements remote_serde: "svd_xkv" deserialization for LMCache v1.
"""

# Standard
import io
from typing import Optional

# Third Party
import numpy as np
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer

logger = init_logger(__name__)


def _unpack_npz(byte_array: bytes):
    """Load tensors packed by the SVD encoder (np.savez_compressed)."""
    buf = io.BytesIO(byte_array)
    npz = np.load(buf)
    # Expecting keys: meta, A_k, B_k, A_v, B_v
    meta = torch.from_numpy(npz["meta"])           # int32 [4] = [L, T, hidden, rank]
    A_k = torch.from_numpy(npz["A_k"])             # [hidden, rank] (fp16)
    B_k = torch.from_numpy(npz["B_k"])             # [L, T, rank]   (fp16)
    A_v = torch.from_numpy(npz["A_v"])             # [hidden, rank] (fp16)
    B_v = torch.from_numpy(npz["B_v"])             # [L, T, rank]   (fp16)
    return meta, A_k, B_k, A_v, B_v


class SvdXKVDeserializer(Deserializer):
    """
    Cross-layer SVD deserializer for LMCache v1.
    Reconstructs full-precision KV chunk in KV_2LTD format: [2, L, T, hidden].
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.dtype = metadata.kv_dtype  # typically torch.float16
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt  # expected "vllm" in this pipeline
        # No need for key/value bins or output buffers here — unlike CacheGen.

    @_lmcache_nvtx_annotate
    def deserialize(self, buffer_memory_obj: BytesBufferMemoryObj) -> MemoryObj:
        """
        Convert the compressed SVD payload back into a KV_2LTD tensor:
        shape = [2, num_layers, num_tokens, hidden]
        """
        # Load & move to device
        meta, A_k, B_k, A_v, B_v = _unpack_npz(buffer_memory_obj.byte_array)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta = meta.to(torch.int32).to(device)
        A_k, B_k, A_v, B_v = (
            A_k.to(device=device, dtype=self.dtype),
            B_k.to(device=device, dtype=self.dtype),
            A_v.to(device=device, dtype=self.dtype),
            B_v.to(device=device, dtype=self.dtype),
        )

        # Parse meta
        nlayer, ntokens, hidden, rank = meta.tolist()

        # Reconstruct K/V:  [L,T,r] @ [r,hidden] -> [L,T,hidden]
        # (We stored A as [hidden, r] and B as [L, T, r] in the encoder.)
        # Use matmul with explicit transpose for clarity.
        K = torch.matmul(B_k, A_k.transpose(0, 1))  # [L, T, hidden]
        V = torch.matmul(B_v, A_v.transpose(0, 1))  # [L, T, hidden]

        # Stack to KV_2LTD (flattened heads): [2, L, T, hidden]
        kv_chunk = torch.stack([K, V], dim=0).to(self.dtype)

        if self.fmt != "vllm":
            # For LMCache v1 + vLLM, tensor format is KV_2LTD flattened hidden.
            # Other fmts can be added if needed.
            raise RuntimeError(f"Unknown format {self.fmt}")

        # Wrap into a TensorMemoryObj in KV_2LTD format
        memory_obj = TensorMemoryObj(
            raw_data=kv_chunk,
            metadata=MemoryObjMetadata(
                shape=kv_chunk.shape,                 # [2, L, T, hidden]
                dtype=kv_chunk.dtype,
                address=-1,
                phy_size=kv_chunk.numel() * kv_chunk.element_size(),
                ref_count=-1,                         # avoid mis-free (parity with CacheGen)
                fmt=MemoryFormat.KV_2LTD,
            ),
            parent_allocator=None,
        )

        logger.info(
            f"[SVD-XKV] decoded KV chunk -> shape={tuple(kv_chunk.shape)}, "
            f"dtype={kv_chunk.dtype}, rank={rank}"
        )
        return memory_obj