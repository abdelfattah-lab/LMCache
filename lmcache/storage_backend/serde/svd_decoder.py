# SPDX-License-Identifier: Apache-2.0
"""
SVD Decoder - Reconstructs KV cache tensors from SVD-compressed data.

This module provides the core SVD decoding functionality that reconstructs
original KV cache tensors from compressed SVD components.
"""
# Standard
from typing import Dict, List

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.serde import Deserializer
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
def svd_decode_single_layer(
    svd_components: Dict[str, torch.Tensor],
    num_heads: int,
    head_size: int,
) -> torch.Tensor:
    """
    Reconstruct a single layer KV tensor from SVD components.
    
    Args:
        svd_components: Dictionary containing:
            - 'U': Left singular vectors [bs, num_tokens, rank]
            - 'S': Singular values [bs, rank]
            - 'Vt': Right singular vectors [bs, rank, hidden_dim]
        num_heads: Number of attention heads
        head_size: Size of each attention head
        
    Returns:
        Reconstructed tensor of shape [num_tokens, num_heads, head_size]
    """
    U_trunc = svd_components['U']  # [bs, num_tokens, rank]
    S_trunc = svd_components['S']  # [bs, rank]
    Vt_trunc = svd_components['Vt']  # [bs, rank, hidden_dim]
    
    # Reconstruct: tensor_reshaped = U_trunc @ diag(S_trunc) @ Vt_trunc
    # [bs, num_tokens, rank] @ [bs, rank, rank] @ [bs, rank, num_heads * head_size] 
    # -> [bs, num_tokens, num_heads * head_size]
    S_diag = torch.diag_embed(S_trunc)  # [bs, rank, rank]
    reconstructed = U_trunc @ S_diag @ Vt_trunc  # [bs, num_tokens, num_heads * head_size]
    
    # Reshape back to [bs, num_tokens, num_heads, head_size]
    bs, num_tokens, _ = reconstructed.shape
    reconstructed = reconstructed.reshape(bs, num_tokens, num_heads, head_size)
    
    # Remove batch dimension: [num_tokens, num_heads, head_size]
    reconstructed = reconstructed.squeeze(0)
    
    return reconstructed


@_lmcache_nvtx_annotate
def decode_function(
    compressed_data: List[Dict[str, torch.Tensor]],
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Decode SVD-compressed data back to full KV cache tensor.
    
    This is the main decoding function that reconstructs the full KV cache tensor
    from compressed SVD components.
    
    Args:
        compressed_data: List of dictionaries containing SVD components,
                        one per (layer, kv_type) combination
        num_layers: Number of transformer layers
        num_tokens: Sequence length
        num_heads: Number of attention heads
        head_size: Size of each attention head
        dtype: Data type for output tensor
        device: Device for output tensor
        
    Returns:
        Reconstructed tensor of shape [num_layers, 2, num_tokens, num_heads, head_size]
    """
    kv_type = 2  # key and value
    expected_components = num_layers * kv_type
    
    if len(compressed_data) != expected_components:
        raise ValueError(
            f"Expected {expected_components} compressed components "
            f"({num_layers} layers * {kv_type} types), "
            f"got {len(compressed_data)}"
        )
    
    # Reconstruct tensor layer by layer
    reconstructed_layers = []
    
    data_idx = 0
    for layer_idx in range(num_layers):
        layer_kv = []
        for kv_idx in range(kv_type):  # 0=key, 1=value
            svd_components = compressed_data[data_idx]
            data_idx += 1
            
            # Reconstruct single layer+type
            reconstructed = svd_decode_single_layer(
                svd_components,
                num_heads,
                head_size,
            )
            layer_kv.append(reconstructed)
        
        # Stack key and value: [2, num_tokens, num_heads, head_size]
        layer_tensor = torch.stack(layer_kv, dim=0)
        reconstructed_layers.append(layer_tensor)
    
    # Stack all layers: [num_layers, 2, num_tokens, num_heads, head_size]
    if len(reconstructed_layers) > 1:
        full_tensor = torch.stack(reconstructed_layers, dim=0)
    else:
        full_tensor = reconstructed_layers[0].unsqueeze(0)
    
    # Ensure correct dtype and device
    full_tensor = full_tensor.to(dtype=dtype, device=device)
    
    logger.debug(
        f"SVD decode_function: Reconstructed {num_layers} layers, {kv_type} types, "
        f"shape={full_tensor.shape}"
    )
    
    return full_tensor


def verify_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Verify that reconstructed tensor is close to original.
    
    Args:
        original: Original tensor
        reconstructed: Reconstructed tensor from SVD
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if tensors are close within tolerance
    """
    return torch.allclose(original, reconstructed, rtol=rtol, atol=atol)


class SVDDeserializer(Deserializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata, dtype):
        self.config = config
        self.metadata = metadata
        self.dtype = dtype
        self.rank = getattr(config, 'svd_rank', None)
        self.num_heads = metadata.kv_shape[0] if metadata.kv_shape else None
        self.head_size = metadata.kv_shape[1] if metadata.kv_shape else None
        self.fmt = metadata.fmt

    @_lmcache_nvtx_annotate
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        """
        Deserialize SVD-compressed bytes back to KV cache tensor.
        
        Args:
            bs: Compressed bytes
        
        Returns:
            Reconstructed tensor of shape [num_layers, 2, num_tokens, num_heads, head_size]
        """
        # Load dict from bytes (torch.save format)
        import io
        buffer = io.BytesIO(bs)
        result = torch.load(buffer, weights_only=False)
        
        # Validate structure
        if 'compressed_data' not in result or 'metadata' not in result:
            raise ValueError("Invalid SVD serialized format: missing 'compressed_data' or 'metadata'")
        
        compressed_data = result['compressed_data']
        meta = result['metadata']
        
        # Extract metadata with validation
        required_keys = ['num_layers', 'num_tokens', 'num_heads', 'head_size']
        for key in required_keys:
            if key not in meta:
                raise ValueError(f"Invalid SVD metadata: missing '{key}'")
        
        num_layers = meta['num_layers']
        num_tokens = meta['num_tokens']
        num_heads = meta['num_heads']
        head_size = meta['head_size']
        dtype = self.dtype
        device = torch.device('cuda')
        
        # Decode
        full_tensor = decode_function(
            compressed_data,
            num_layers,
            num_tokens,
            num_heads,
            head_size,
            dtype,
            device,
        )
        
        # full_tensor is [num_layers, 2, num_tokens, num_heads, head_size]
        # Return format based on metadata.fmt
        if self.fmt == "vllm":
            return full_tensor
        elif self.fmt == "huggingface":
            # [num_layers, 2, num_tokens, num_heads, head_size] -> [num_layers, 2, num_heads, num_tokens, head_size]
            return full_tensor.permute(0, 1, 3, 2, 4)
        else:
            raise RuntimeError(f"Unknown format {self.fmt}")
