# SPDX-License-Identifier: Apache-2.0
"""
SVD-XKV Encoder (v1-style)
Implements remote_serde: "svd_xkv" for LMCache v1
"""

# Standard
import io
import torch
import numpy as np

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import BytesBufferMemoryObj, MemoryObj
from lmcache.v1.storage_backend.naive_serde.serde import Serializer

logger = init_logger(__name__)


def _split_kv(tensor: torch.Tensor):
    """
    Split [num_layers, 2, num_tokens, num_heads, head_size]
    → (K, V) with shape [num_layers, num_tokens, hidden]
    """
    nlayer, _, ntok, nhead, hsize = tensor.shape
    hidden = nhead * hsize
    kv = tensor.reshape(nlayer, 2, ntok, hidden)
    return kv[:, 0], kv[:, 1]


def _pack_tensors_to_bytes(tensors: dict) -> bytes:
    """Serialize tensors to bytes with numpy npz format"""
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v.cpu().numpy() for k, v in tensors.items()})
    return buf.getvalue()

class SvdXKVSerializer(Serializer):
    """
    Cross-layer SVD serializer for LMCache v1 (uses MemoryObj API).
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.chunk_size = config.chunk_size
        self.svd_ratio = getattr(config, "svd_ratio", 0.2)
        self.svd_rank = getattr(config, "svd_rank", None)
        self.fmt = metadata.fmt
        self.kv_shape = metadata.kv_shape
        self.dtype = torch.float16

    @_lmcache_nvtx_annotate
    def serialize(self, memory_obj: MemoryObj) -> BytesBufferMemoryObj:
        """
        Serialize a KV_2LTD MemoryObj to SVD_XKV compressed MemoryObj.
        """
        assert memory_obj.tensor is not None, "MemoryObj has no tensor."
        tensor = memory_obj.tensor.cuda()

        # Ensure device alignment
        if torch.cuda.current_device != tensor.device:
            torch.cuda.set_device(tensor.device)

        # tensor: [2, num_layers, num_tokens, hidden_size]
        tensor = tensor.view(*tensor.shape[:-1], self.kv_shape[-2], self.kv_shape[-1])
        tensor = tensor.permute([1, 0, 2, 3, 4])  # [num_layers, 2, num_tokens, num_heads, head_size]

        # Split K/V
        fp_k, fp_v = _split_kv(tensor)
        nlayer, ntok, hidden = fp_k.shape
        rank = self.svd_rank or int(hidden * self.svd_ratio)
        device = tensor.device

        # Perform cross-layer SVD
        def compress_block(X: torch.Tensor, rank: int):
            Xmat = X.reshape(-1, hidden).float()  # [nlayer * ntok, hidden]
            U, S, Vt = torch.linalg.svd(Xmat, full_matrices=False)
            A = Vt[:rank, :].T  # shared basis
            B = (U[:, :rank] * S[:rank]).reshape(nlayer, ntok, rank)
            return A.to(self.dtype), B.to(self.dtype)

        A_k, B_k = compress_block(fp_k, rank)
        A_v, B_v = compress_block(fp_v, rank)

        # Package as bytes
        payload = _pack_tensors_to_bytes({
            "meta": torch.tensor([nlayer, ntok, hidden, rank], dtype=torch.int32),
            "A_k": A_k, "B_k": B_k,
            "A_v": A_v, "B_v": B_v,
        })

        compressed_size = len(payload) / 1e6
        orig_size = tensor.numel() * tensor.element_size() / 1e6
        ratio = orig_size / compressed_size
        logger.info(
            f"[SVD-XKV] layers={nlayer} tokens={ntok} rank={rank} "
            f"orig={orig_size:.2f}MB comp={compressed_size:.2f}MB ratio={ratio:.2f}x"
        )

        return BytesBufferMemoryObj(payload)