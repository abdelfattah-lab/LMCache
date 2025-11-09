# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Tuple, Dict
import io
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.serde import Serializer
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


def _split_kv(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a blob KV tensor into K and V with merged heads.

    Input:
        tensor: [num_layers, 2, num_tokens, num_heads, head_size]

    Returns:
        K, V: [num_layers, num_tokens, hidden] with hidden=num_heads*head_size
    """
    nlayer, _, ntok, nhead, hsize = tensor.shape
    hidden = nhead * hsize
    kv = tensor.reshape(nlayer, 2, ntok, hidden)
    return kv[:, 0], kv[:, 1]


def _torch_serialize_to_bytes(obj: Dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    # Use torch.save so tensors keep dtype/device info when reasonable.
    # (They will be reloaded to CPU first; we move to GPU explicitly later.)
    torch.save(obj, buf)
    return buf.getvalue()


def _torch_load_from_bytes(payload: bytes) -> Dict[str, torch.Tensor]:
    buf = io.BytesIO(payload)
    obj = torch.load(buf, map_location="cpu")
    return obj


class SvdXKVSerializer(Serializer):
    """
    Cross-layer SVD serializer for LMCache.
    Implements remote_serde: "svd_xkv"

    - Compresses each incoming chunk’s KV
    - Stores (meta, A_k, B_k, A_v, B_v) in bytes
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt  # e.g., "huggingface"
        # Rank selection: prefer explicit rank; else ratio of hidden dim (default 0.2)
        # You can plumb these via YAML: svd_rank, svd_ratio
        self.svd_rank = getattr(config, "svd_rank", None)
        self.svd_ratio = float(getattr(config, "svd_ratio", 0.2))
        # Output dtype for factors (fp16 keeps size down, good enough for recon)
        self.dtype = torch.float16

    @_lmcache_nvtx_annotate
    def to_bytes(self, tensor: torch.Tensor) -> bytes:
        """
        Serialize a torch KV tensor (device can be CPU or CUDA) to a compressed SVD payload.

        Input:
            tensor: [num_layers, 2, num_tokens, num_heads, head_size]
        Returns:
            bytes
        """
        if tensor.ndim != 5:
            raise ValueError(f"SVD serializer expected 5D KV, got {tuple(tensor.shape)}")

        # Align device (mirror CacheGen’s pattern of working on the input device)
        dev = tensor.device
        if torch.cuda.is_available() and dev.type != "cuda":
            # If model chunks come on CPU we can choose to move to GPU for speed.
            tensor = tensor.cuda()

        # Match CacheGen’s permute handling
        # CacheGen expects [L, 2, T, H, D] after permute for 'huggingface'
        if self.fmt == "huggingface":
            tensor = tensor.permute(0, 1, 3, 2, 4)

        # Expect [L, 2, T, H, D]
        L, two, T, H, D = tensor.shape
        assert two == 2, "2 stands for (K,V)"

        hidden = H * D

        # Merge heads and split K/V → [L, T, hidden]
        fp_k, fp_v = _split_kv(tensor)

        # Flatten (L,T,hidden) → (LT, hidden) for a single SVD per stream
        # (This matches "cross-layer" by spanning all layers in this chunk.)
        def svd_compress(X: torch.Tensor, rank_hint: int) -> Tuple[torch.Tensor, torch.Tensor]:
            Xmat = X.reshape(-1, hidden).float()  # (L*T, hidden)
            # Use torch.linalg.svd on the same device as Xmat
            U, S, Vh = torch.linalg.svd(Xmat, full_matrices=False)
            # Pick rank
            r = rank_hint
            # Shared basis (hidden x r); coefficient (L*T x r)
            A = Vh[:r, :].T                         # (hidden, r)
            B = U[:, :r] * S[:r]                    # (L*T, r)
            return A.to(self.dtype), B.to(self.dtype)

        # Pick rank: if not provided, use svd_ratio * hidden (at least 1)
        rank_k = self.svd_rank or max(1, int(self.svd_ratio * hidden))
        rank_v = self.svd_rank or max(1, int(self.svd_ratio * hidden))

        t0 = time.time()
        A_k, B_k = svd_compress(fp_k, rank_k)
        A_v, B_v = svd_compress(fp_v, rank_v)
        t1 = time.time()

        # Pack metadata + tensors
        meta = torch.tensor([L, T, H, D, rank_k, rank_v], dtype=torch.int32)
        payload = _torch_serialize_to_bytes(
            {
                "meta": meta,        # [L, T, H, D, rk, rv]
                "A_k": A_k,          # [hidden, rk]
                "B_k": B_k,          # [L*T, rk]
                "A_v": A_v,          # [hidden, rv]
                "B_v": B_v,          # [L*T, rv]
            }
        )

        orig_bytes = (L * 2 * T * H * D) * tensor.element_size()
        ratio = orig_bytes / max(len(payload), 1)
        logger.info(
            f"[SVD-XKV] Encoded chunk: L={L} T={T} H={H} D={D} "
            f"rk={rank_k} rv={rank_v} "
            f"size={len(payload)/1e6:.2f}MB ratio={ratio:.2f}x time={(t1 - t0):.3f}s"
        )
        return payload

    @_lmcache_nvtx_annotate
    def from_bytes(self, payload: bytes) -> torch.Tensor:
        """
        Deserialize SVD payload back into a KV tensor on the current device.

        Returns:
            KV tensor of shape [num_layers, 2, num_tokens, num_heads, head_size]
            (with format adjusted by metadata.fmt, mirroring CacheGenSerializer)
        """
        t0 = time.time()
        obj = _torch_load_from_bytes(payload)

        meta: torch.Tensor = obj["meta"].to(torch.int32)
        L, T, H, D, rk, rv = meta.tolist()
        hidden = H * D

        # Move to CUDA if available (match encode side)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        A_k = obj["A_k"].to(device=device, dtype=self.dtype)   # [hidden, rk]
        B_k = obj["B_k"].to(device=device, dtype=self.dtype)   # [L*T, rk]
        A_v = obj["A_v"].to(device=device, dtype=self.dtype)   # [hidden, rv]
        B_v = obj["B_v"].to(device=device, dtype=self.dtype)   # [L*T, rv]

        # Reconstruct [L*T, hidden] then reshape → [L, T, hidden]
        K = (B_k @ A_k.T).reshape(L, T, hidden)
        V = (B_v @ A_v.T).reshape(L, T, hidden)

        # Restore heads and KV axis
        K = K.reshape(L, T, H, D)
        V = V.reshape(L, T, H, D)
        kv = torch.stack([K, V], dim=1)  # [L, 2, T, H, D]

        # Reverse the earlier permute if needed
        if self.fmt == "huggingface":
            kv = kv.permute(0, 1, 3, 2, 4)  # back to [L, 2, H, T, D] if required by caller

        kv = kv.to(self.dtype)
        t1 = time.time()
        logger.info(
            f"[SVD-XKV] Decoded chunk: L={L} T={T} H={H} D={D} rk={rk} rv={rv} time={(t1 - t0):.3f}s"
        )
        return kv