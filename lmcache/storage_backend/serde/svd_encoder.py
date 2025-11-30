# SPDX-License-Identifier: Apache-2.0
"""
SVD Encoder - Performs SVD compression on KV cache tensors.

This module provides the core SVD encoding functionality that compresses
KV cache tensors by performing Singular Value Decomposition and truncating
to a specified rank.
"""
# Standard
from typing import Dict, List, Tuple

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.serde import Serializer
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
def svd_encode_single_layer(
    tensor: torch.Tensor,
    rank: int,
) -> Dict[str, torch.Tensor]:
    """
    Perform SVD compression on a single layer KV tensor.
    
    Args:
        tensor: Input tensor of shape [num_tokens, num_heads, head_size]
        rank: Number of singular values to keep
        
    Returns:
        Dictionary containing:
            - 'U': Left singular vectors [1, num_tokens, rank]
            - 'S': Singular values [1, rank]
            - 'Vt': Right singular vectors [1, rank, num_heads*head_size]
    """
    # Prepare for SVD following standard pattern:
    # [num_tokens, num_heads, head_size] -> [1, num_tokens, num_heads*head_size]
    bs = 1
    num_tokens, num_heads, head_size = tensor.shape
    
    # Add batch dimension: [num_tokens, num_heads, head_size] -> [1, num_tokens, num_heads, head_size]
    x = tensor.unsqueeze(0)
    
    # Reshape: [1, num_tokens, num_heads, head_size] -> [1, num_tokens, num_heads*head_size]
    x2d = x.reshape(bs, num_tokens, num_heads * head_size)
    
    # Convert to float32 for SVD stability
    original_dtype = x2d.dtype
    if original_dtype in (torch.bfloat16, torch.float16):
        x2d = x2d.to(torch.float32)
    
    # Perform SVD
    U, S, Vh = torch.linalg.svd(x2d, full_matrices=False)
    
    # Determine actual rank to use
    r = min(rank, S.shape[-1])
    
    # Truncate to rank
    U_trunc = U[:, :, :r]  # [bs, num_tokens, r]
    S_trunc = S[:, :r]      # [bs, r]
    Vh_trunc = Vh[:, :r, :]  # [bs, r, num_heads*head_size]
    
    # Convert back to original dtype if needed
    if original_dtype in (torch.bfloat16, torch.float16):
        U_trunc = U_trunc.to(original_dtype)
        S_trunc = S_trunc.to(original_dtype)
        Vh_trunc = Vh_trunc.to(original_dtype)
    
    return {
        'U': U_trunc,
        'S': S_trunc,
        'Vt': Vh_trunc,
    }


@_lmcache_nvtx_annotate
def encode_function(
    kv: torch.Tensor,
    rank: int,
    num_heads: int,
    head_size: int,
) -> List[Dict[str, torch.Tensor]]:
    """
    Encode KV cache tensor using SVD compression.
    
    This is the main encoding function that processes the full KV cache tensor
    and compresses each layer and key/value type separately.
    
    Args:
        kv: Input tensor of shape [num_layers, 2, num_tokens, num_heads, head_size]
            where dimension 1 represents [key, value]
        rank: Number of singular values to keep for compression
        num_heads: Number of attention heads
        head_size: Size of each attention head
        
    Returns:
        List of dictionaries, one per (layer, kv_type) combination.
        Each dictionary contains compressed SVD components (U, S, Vt).
    """
    num_layers = kv.shape[0]
    kv_type = kv.shape[1]  # Should be 2 (key and value)
    
    if kv_type != 2:
        raise ValueError(f"Expected kv_type=2 (key, value), got {kv_type}")
    
    compressed_data = []
    
    # Process each layer and kv_type (key/value) separately
    for layer_idx in range(num_layers):
        for kv_idx in range(kv_type):  # 0=key, 1=value
            # Extract single layer+type: [num_tokens, num_heads, head_size]
            layer_tensor = kv[layer_idx, kv_idx, :, :, :]
            
            # Compress using SVD
            svd_components = svd_encode_single_layer(layer_tensor, rank)
            compressed_data.append(svd_components)
    
    logger.debug(
        f"SVD encode_function: Compressed {num_layers} layers, {kv_type} types, "
        f"rank={rank}, output={len(compressed_data)} components"
    )
    
    return compressed_data


def get_compressed_size(
    num_tokens: int,
    hidden_dim: int,
    rank: int,
) -> Tuple[int, int]:
    """
    Calculate the compressed size after SVD compression.
    
    Args:
        num_tokens: Sequence length
        hidden_dim: Hidden dimension (num_heads * head_size)
        rank: SVD rank
        
    Returns:
        Tuple of (original_size, compressed_size) in number of float elements
    """
    original_size = num_tokens * hidden_dim
    # U: [num_tokens, rank], S: [rank], Vt: [rank, hidden_dim]
    compressed_size = num_tokens * rank + rank + rank * hidden_dim
    
    return original_size, compressed_size


class SVDSerializer(Serializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata, dtype):
        self.config = config
        self.metadata = metadata
        self.dtype = dtype
        # Get SVD rank from config
        self.rank = getattr(config, 'svd_rank', None)
        if self.rank is None:
            # Default to half of hidden dimension
            if metadata.kv_shape:
                hidden_dim = metadata.kv_shape[0] * metadata.kv_shape[1]
                self.rank = hidden_dim // 2
            else:
                raise ValueError("svd_rank not specified in config and kv_shape not available")
        self.num_heads = metadata.kv_shape[0] if metadata.kv_shape else None
        self.head_size = metadata.kv_shape[1] if metadata.kv_shape else None
        self.fmt = metadata.fmt

    @_lmcache_nvtx_annotate
    def to_bytes(self, tensor: torch.Tensor) -> bytes:
        """
        Serialize a KV cache tensor using SVD compression.
        
        Args:
            tensor: Input tensor of shape [num_layers, 2, num_tokens, num_heads, head_size]
        
        Returns:
            Compressed bytes
        """
        # Compress using SVD
        compressed_data = encode_function(
            tensor,
            self.rank,
            self.num_heads,
            self.head_size,
        )
        
        # Package all data
        result = {
            'compressed_data': compressed_data,
            'metadata': {
                'num_layers': tensor.shape[0],
                'num_tokens': tensor.shape[2],
                'num_heads': self.num_heads,
                'head_size': self.head_size,
                'rank': self.rank,
                'dtype': str(self.dtype),
                'fmt': self.fmt,
            }
        }
        
        # Serialize to bytes using torch.save
        import io
        buffer = io.BytesIO()
        torch.save(result, buffer)
        return buffer.getvalue()
