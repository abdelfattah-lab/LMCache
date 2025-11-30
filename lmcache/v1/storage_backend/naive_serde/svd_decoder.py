# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.svd_decoder import decode_function
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import BytesBufferMemoryObj, MemoryObj, TensorMemoryObj
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer
from lmcache.v1.memory_management import MemoryObjMetadata, MemoryFormat

logger = init_logger(__name__)


class SVDDeserializer(Deserializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        self.kv_shape = metadata.kv_shape
        self.dtype = metadata.kv_dtype
        
        logger.info(f"SVDDeserializer initialized with dtype={self.dtype}")

    @_lmcache_nvtx_annotate
    def deserialize(self, memory_obj: MemoryObj, layer_id: Optional[int] = None) -> MemoryObj:
        """
        Deserialize SVD-compressed data back to tensor.
        
        Input:
            memory_obj: BytesBufferMemoryObj containing compressed SVD data
            layer_id: Optional layer ID for layerwise mode
            
        Returns:
            TensorMemoryObj with reconstructed tensor
        """
        assert isinstance(memory_obj, BytesBufferMemoryObj)
        
        # Deserialize from bytes
        result = self._from_bytes(memory_obj.byte_array)
        
        compressed_data = result['compressed_data']
        meta = result['metadata']
        
        logger.info(
            f"SVDDeserializer.deserialize: reconstructing shape={meta['original_shape']}, "
            f"rank={meta['rank']}, layer_id={layer_id}"
        )
        
        num_tokens = meta['num_tokens']
        num_heads = meta['num_heads']
        head_size = meta['head_size']
        
        # Use the decoder function to reconstruct the single layer
        # decode_function returns shape: [2, num_tokens, num_heads, head_size]
        single_layer = decode_function(
            compressed_data,
            num_tokens,
            num_heads,
            head_size,
            self.dtype,
            torch.device('cuda'),
        )
        
        # single_layer is [2, num_tokens, num_heads, head_size]
        # Add layer dimension: [2, 1, num_tokens, num_heads, head_size]
        full_tensor = single_layer.unsqueeze(1)
        
        # Reshape to [2, 1, num_tokens, hidden_size]
        hidden_size = num_heads * head_size
        full_tensor = full_tensor.reshape(2, 1, num_tokens, hidden_size)
        
        # Convert back to original format if needed
        original_shape = tuple(meta['original_shape'])
        if len(original_shape) == 3:
            # Layerwise format: convert back to [num_tokens, 2, hidden_size]
            full_tensor = full_tensor.squeeze(1)  # Remove layer dim: [2, num_tokens, hidden_size]
            full_tensor = full_tensor.permute([1, 0, 2])  # -> [num_tokens, 2, hidden_size]
        
        logger.info(
            f"SVDDeserializer.deserialize: reconstructed to shape={full_tensor.shape}"
        )
    
        metadata = MemoryObjMetadata(
            shape=full_tensor.shape,
            dtype=self.dtype,
            address=-1,
            phy_size=full_tensor.numel() * full_tensor.element_size(),
            ref_count=-1, 
            fmt=MemoryFormat.KV_T2D,
        )
        
        return TensorMemoryObj(
            raw_data=full_tensor,
            metadata=metadata,
            parent_allocator=None,
        )

    def _from_bytes(self, bytes_data: bytes) -> dict:
        """Load result dict from bytes using torch.load."""
        import io
        buffer = io.BytesIO(bytes_data)
        return torch.load(buffer, weights_only=False)
