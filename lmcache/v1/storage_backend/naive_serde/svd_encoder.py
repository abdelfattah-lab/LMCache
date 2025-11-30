# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.svd_encoder import encode_function
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import BytesBufferMemoryObj, MemoryObj
from lmcache.v1.storage_backend.naive_serde.serde import Serializer

logger = init_logger(__name__)


class SVDSerializer(Serializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        self.kv_shape = metadata.kv_shape
        self.dtype = metadata.kv_dtype
        
        # Get SVD rank from config, default to half of hidden dimension
        hidden_dim = self.kv_shape[-2] * self.kv_shape[-1]  # num_heads * head_size
        default_rank = hidden_dim // 2
        self.rank = config.get_extra_config_value("svd_rank", default_rank)
        
        logger.info(
            f"SVDSerializer initialized with rank={self.rank}, "
            f"hidden_dim={hidden_dim}, dtype={self.dtype}"
        )

    @_lmcache_nvtx_annotate
    def serialize(self, memory_obj: MemoryObj, layer_id: Optional[int] = None) -> BytesBufferMemoryObj:
        """
        Serialize a MemoryObj using SVD compression.
        
        Input:
            memory_obj: the memory object to be serialized. Expected shape:
                - Layerwise: [num_tokens, 2, hidden_size] (KV_T2D format)
                - Non-layerwise: [2, num_layers, num_tokens, hidden_size]
            layer_id: Optional layer ID for layerwise mode.

        Returns:
            BytesBufferMemoryObj: Compressed data containing U_trunc, S_trunc, Vt_trunc,
                                  and metadata for reconstruction.
        """
        assert memory_obj.tensor is not None
        tensor = memory_obj.tensor.cuda()
        
        logger.info(
            f"SVDSerializer.serialize: input shape={tensor.shape}, "
            f"format={memory_obj.metadata.fmt}, layer_id={layer_id}, "
            f"dtype={memory_obj.metadata.dtype}"
        )

        # Set device
        if torch.cuda.current_device != tensor.device:
            torch.cuda.set_device(tensor.device)

        # Handle sparse tensors
        if tensor.is_sparse:
            logger.warning(f"Converting sparse tensor to dense, shape={tensor.shape}")
            tensor = tensor.to_dense()

        # Ensure contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Process different tensor formats
        num_dims = len(tensor.shape)
        original_shape = tensor.shape
        
        if num_dims == 3:
            # Layerwise format: [num_tokens, 2, hidden_size] (KV_T2D)
            if tensor.shape[1] == 2:
                # Correct KV_T2D format
                # Permute to [2, num_tokens, hidden_size]
                tensor = tensor.permute([1, 0, 2])
            elif tensor.shape[0] == 2:
                # Legacy format: [2, num_tokens, hidden_size]
                logger.warning(f"Legacy format detected: {tensor.shape}")
                pass  # Already in correct format
            else:
                raise ValueError(f"Unexpected 3D tensor shape {tensor.shape}")
                
            # Add layer dimension: [2, 1, num_tokens, hidden_size]
            tensor = tensor.unsqueeze(1)
            
        elif num_dims == 4:
            # Multi-layer format: [2, num_layers, num_tokens, hidden_size]
            pass
        else:
            raise ValueError(
                f"Expected 3D or 4D tensor, got shape {tensor.shape}"
            )

        # At this point: tensor is [2, num_layers, num_tokens, hidden_size]
        kv_type, num_layers, num_tokens, hidden_size = tensor.shape
        
        # Validate hidden_size
        expected_hidden_size = self.kv_shape[-2] * self.kv_shape[-1]
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"Hidden size mismatch: got {hidden_size}, "
                f"expected {expected_hidden_size}"
            )

        # Reshape to [2, num_layers, num_tokens, num_heads, head_size]
        tensor = tensor.view(kv_type, num_layers, num_tokens, 
                           self.kv_shape[-2], self.kv_shape[-1])
        
        # For layerwise mode, extract specific layer if needed
        if layer_id is not None and num_layers > 1:
            logger.warning(
                f"Layerwise mode: extracting layer {layer_id} from "
                f"{num_layers} layers"
            )
            tensor = tensor[:, layer_id:layer_id+1, :, :, :]
            num_layers = 1

        # Permute to [num_layers, 2, num_tokens, num_heads, head_size]
        tensor = tensor.permute([1, 0, 2, 3, 4])
        
        # In layerwise mode, we should only have 1 layer
        # If we have more than 1 layer, extract the requested layer
        if num_layers > 1:
            if layer_id is not None:
                logger.warning(
                    f"Layerwise mode: Received tensor with {num_layers} layers but layer_id={layer_id}. "
                    f"Extracting layer {layer_id} from multi-layer tensor."
                )
                tensor = tensor[layer_id:layer_id+1, :, :, :, :]  # [1, 2, num_tokens, num_heads, head_size]
                num_layers = 1
            else:
                raise ValueError(
                    f"Single-layer SVD received {num_layers} layers but no layer_id specified. "
                    f"Cannot determine which layer to compress."
                )
        
        # Extract single layer: [2, num_tokens, num_heads, head_size]
        single_layer = tensor[0]  # Remove layer dimension

        # Use the encoder function to perform SVD compression (single layer)
        compressed_data = encode_function(
            single_layer,
            self.rank,
            self.kv_shape[-2],  # num_heads
            self.kv_shape[-1],  # head_size
        )
        
        # Package all data
        result = {
            'compressed_data': compressed_data,
            'metadata': {
                'original_shape': list(original_shape),
                'num_tokens': num_tokens,
                'num_heads': self.kv_shape[-2],
                'head_size': self.kv_shape[-1],
                'rank': self.rank,
                'dtype': str(self.dtype),
                'layer_id': layer_id,
            }
        }
        
        # Serialize to bytes
        bytes_data = self._to_bytes(result)
        
        logger.info(
            f"SVDSerializer.serialize: compressed {original_shape} to "
            f"{len(bytes_data)} bytes (rank={self.rank})"
        )
        
        return BytesBufferMemoryObj(bytes_data)

    def _to_bytes(self, result: dict) -> bytes:
        """Convert result dict to bytes using torch.save."""
        import io
        buffer = io.BytesIO()
        torch.save(result, buffer)
        return buffer.getvalue()
