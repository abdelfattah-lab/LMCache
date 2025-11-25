# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_encoder import encode_function
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import BytesBufferMemoryObj, MemoryObj
from lmcache.v1.storage_backend.naive_serde.cachegen_basics import CacheGenConfig
from lmcache.v1.storage_backend.naive_serde.serde import Serializer

logger = init_logger(__name__)


class CacheGenSerializer(Serializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

        self.kv_shape = metadata.kv_shape

    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.kspecs:
            ret[spec.start_layer : spec.end_layer] = spec.bins
        return ret.cuda()

    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.vspecs:
            ret[spec.start_layer : spec.end_layer] = spec.bins
        return ret.cuda()

    # TODO(Jiayi): A lot of memory copies can be avoided in this function.
    @_lmcache_nvtx_annotate
    def serialize(self, memory_obj: MemoryObj, layer_id: Optional[int] = None) -> BytesBufferMemoryObj:
        """
        Serialize a KV_2LTD MemoryObj to CACHEGEN_BINARY MemoryObj.

        Input:
            memory_obj: the memory object to be serialized.

        Returns:
            MemoryObj: the serialized binary memory object.
        """

        # TODO(Jiayi): please avoid this copy by directly performing
        # serialization inside gpu connector.
        assert memory_obj.tensor is not None
        tensor = memory_obj.tensor.cuda()

        # Temporary fix for issue #83: encoder will have the default device 0
        # on all the ray workers. Need to set it to the correct device.
        # Also need to figure out why this happens.
        if torch.cuda.current_device != tensor.device:
            torch.cuda.set_device(tensor.device)
        if tensor.device != self.key_bins.device:
            self.key_bins = self.key_bins.to(tensor.device)
        if tensor.device != self.value_bins.device:
            self.value_bins = self.value_bins.to(tensor.device)

        # Handle sparse tensors: convert to dense if needed
        # Sparse tensors cannot use view() or permute() operations
        if tensor.is_sparse:
            logger.warning(
                f"Received sparse tensor with shape {tensor.shape}, converting to dense. "
                f"Expected KV_2LTD format: [2, num_layers, num_tokens, hidden_size]"
            )
            tensor = tensor.to_dense()

        # Ensure tensor is contiguous for view operation
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Handle different tensor formats:
        # 1. Multi-layer format: [2, num_layers, num_tokens, hidden_size] (4 dims) - from non-layerwise connectors
        # 2. Layerwise format: [2, num_tokens, hidden_size] or [num_tokens, 2, hidden_size] (3 dims) - from layerwise connectors
        num_dims = len(tensor.shape)
        
        if num_dims == 3:
            # Layerwise format: single layer tensor
            # Handle both [2, num_tokens, hidden_size] and [num_tokens, 2, hidden_size]
            if tensor.shape[0] == 2:
                # Format: [2, num_tokens, hidden_size]
                # Add layer dimension: [2, 1, num_tokens, hidden_size]
                tensor = tensor.unsqueeze(1)
            elif tensor.shape[1] == 2:
                # Format: [num_tokens, 2, hidden_size]
                # Permute to [2, num_tokens, hidden_size], then add layer dimension: [2, 1, num_tokens, hidden_size]
                tensor = tensor.permute([1, 0, 2])
                tensor = tensor.unsqueeze(1)
            else:
                raise ValueError(
                    f"Unexpected 3D tensor shape {tensor.shape}. Expected either "
                    f"[2, num_tokens, hidden_size] or [num_tokens, 2, hidden_size] for layerwise format."
                )
        elif num_dims == 4:
            # Multi-layer format: [2, num_layers, num_tokens, hidden_size]
            # This is the expected format, no transformation needed
            pass
        else:
            raise ValueError(
                f"Expected 3D (layerwise) or 4D (multi-layer) tensor, "
                f"but got shape {tensor.shape} with {num_dims} dimensions. "
                f"This suggests a mismatch between vLLM output format and LMCache expected format."
            )

        # At this point, tensor should be [2, num_layers, num_tokens, hidden_size]
        # Validate that hidden_size can be split into num_heads * head_size
        hidden_size = tensor.shape[-1]
        expected_hidden_size = self.kv_shape[-2] * self.kv_shape[-1]  # num_heads * head_size
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"Tensor hidden_size ({hidden_size}) does not match expected "
                f"num_heads * head_size ({self.kv_shape[-2]} * {self.kv_shape[-1]} = {expected_hidden_size}). "
                f"kv_shape from metadata: {self.kv_shape}, tensor shape: {tensor.shape}"
            )

        # tensor is [2, num_layers, num_tokens, hidden_size]
        # Reshape to [2, num_layers, num_tokens, num_heads, head_size]
        tensor = tensor.view(*tensor.shape[:-1], self.kv_shape[-2], self.kv_shape[-1])
        # Permute to [num_layers, 2, num_tokens, num_heads, head_size]
        tensor = tensor.permute([1, 0, 2, 3, 4])

        # TODO(Jiayi): remove hardcoded "2"
        """ expecting a tensor of shape 
        [num_layers, 2, num_tokens, num_heads, head_size] """
        ntokens = tensor.shape[2]
        num_layers = tensor.shape[0]
        
        # For layerwise mode (single layer), slice bins to the specific layer
        # If layer_id is provided, use bins for that layer; otherwise use layer 0
        if num_layers == 1 and layer_id is not None:
            key_bins_to_use = self.key_bins[layer_id:layer_id+1]  # Shape [1]
            value_bins_to_use = self.value_bins[layer_id:layer_id+1]  # Shape [1]
        elif num_layers == 1:
            # Single layer but no layer_id provided, use layer 0's bins as fallback
            key_bins_to_use = self.key_bins[0:1]  # Shape [1]
            value_bins_to_use = self.value_bins[0:1]  # Shape [1]
        else:
            # Multi-layer: use bins for all layers
            key_bins_to_use = self.key_bins[:num_layers]  # Shape [num_layers]
            value_bins_to_use = self.value_bins[:num_layers]  # Shape [num_layers]
        
        output_dict = encode_function(
            tensor,
            self.cachegen_config,
            key_bins_to_use,
            value_bins_to_use,
            ntokens,
        )

        return BytesBufferMemoryObj(output_dict.to_bytes())
