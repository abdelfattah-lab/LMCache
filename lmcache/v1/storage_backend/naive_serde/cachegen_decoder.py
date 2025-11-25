# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_basics import (
    CacheGenGPUEncoderOutput,
)
from lmcache.storage_backend.serde.cachegen_decoder import (
    decode_function_gpu,
    do_dequantize,
)
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.naive_serde.cachegen_basics import CacheGenConfig
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer

logger = init_logger(__name__)


class CacheGenDeserializer(Deserializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.dtype = metadata.kv_dtype
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.output_buffer: Optional[torch.Tensor] = None
        self.fmt = metadata.fmt
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

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

    def get_output_buffer(self, nlayers: int, nchannels: int, ntokens: int):
        if (
            self.output_buffer is None
            or self.output_buffer.shape[1] != 2 * nlayers * nchannels
        ):
            self.output_buffer = torch.zeros(
                (self.chunk_size, 2 * nlayers * nchannels), dtype=torch.uint8
            ).cuda()
        return self.output_buffer[:ntokens, :]

    # TODO(Jiayi): A lot of memory copies can be avoided in this function.
    @_lmcache_nvtx_annotate
    def deserialize(self, buffer_memory_obj: BytesBufferMemoryObj, layer_id: Optional[int] = None) -> MemoryObj:
        logger.info(
            f"CacheGenDeserializer.deserialize: RETRIEVING with layer_id={layer_id}, "
            f"buffer_size={len(buffer_memory_obj.byte_array)} bytes"
        )
        encoder_output = CacheGenGPUEncoderOutput.from_bytes(
            buffer_memory_obj.byte_array
        )

        encoder_output.max_tensors_key = encoder_output.max_tensors_key.cuda()
        encoder_output.max_tensors_value = encoder_output.max_tensors_value.cuda()

        ntokens = encoder_output.max_tensors_key.shape[1]
        layers_in_key = encoder_output.max_tensors_key.shape[0]
        key, value = decode_function_gpu(
            encoder_output.cdf,
            encoder_output.data_chunks,
            layers_in_key,
            ntokens,
            self.get_output_buffer(
                encoder_output.cdf.shape[0] // 2,
                encoder_output.cdf.shape[1],
                ntokens,
            ),
        )

        # Temporary fix for #83: change the device of key_bins and value_bins
        # to the device of key and value
        # This requires a long-term fix in the future. Currently,
        # CacheGenGPUEncoderOutput has implicit device in itself.
        # More specifically, if the encoder encodes the tensor on GPU0, the
        # from_bytes will also return a tensor on GPU0
        # We may want to dynamically configure the device based on config and
        # metadata in the future
        if self.key_bins.device != key.device:
            self.key_bins = self.key_bins.to(key.device)

        if self.value_bins.device != value.device:
            self.value_bins = self.value_bins.cuda()

        key = do_dequantize(key, self.key_bins, encoder_output.max_tensors_key)
        value = do_dequantize(value, self.value_bins, encoder_output.max_tensors_value)
        """ merge key and value back and reshape """
        nlayers, ntokens, nchannels = key.shape
        blob = torch.stack([key, value])  # [2, nlayers, ntokens, nchannels]
        blob = blob.reshape(
            (
                2,
                nlayers,
                ntokens,
                encoder_output.num_heads,
                encoder_output.head_size,
            )
        )
        match self.fmt:
            case "vllm":
                hidden_dim = blob.shape[-1] * blob.shape[-2]
                kv_chunk = blob.reshape(*blob.shape[:-2], hidden_dim).to(
                    self.dtype
                )  # [2, nlayers, ntokens, hidden_dim]
            case _:
                raise RuntimeError("Unknown format %s" % self.fmt)

        # For layerwise mode, extract the specific layer and convert to KV_T2D format
        # KV_2LTD: [2, num_layers, num_tokens, hidden_dim]
        # KV_T2D: [num_tokens, 2, hidden_dim] (for single layer)
        if layer_id is not None and nlayers > 1:
            # Layerwise mode: extract the specific layer from multi-layer data
            # This can happen when data was stored in non-layerwise mode but retrieved in layerwise mode
            if layer_id >= nlayers:
                raise ValueError(
                    f"layer_id ({layer_id}) is out of range for decoded data with {nlayers} layers. "
                    f"Shape: {kv_chunk.shape}"
                )
            # Extract the specific layer: [2, nlayers, ntokens, hidden_dim] -> [2, ntokens, hidden_dim] -> [ntokens, 2, hidden_dim]
            kv_chunk = kv_chunk[:, layer_id, :, :]  # Select layer_id from dimension 1 -> [2, ntokens, hidden_dim]
            kv_chunk = kv_chunk.permute([1, 0, 2])  # Permute to [ntokens, 2, hidden_dim]
            output_fmt = MemoryFormat.KV_T2D
            logger.debug(
                f"Extracted layer {layer_id} from multi-layer data (nlayers={nlayers}). "
                f"Original shape: {blob.shape}, extracted shape: {kv_chunk.shape}"
            )
        elif nlayers == 1:
            # Single layer already: [2, 1, ntokens, hidden_dim] -> [2, ntokens, hidden_dim] -> [ntokens, 2, hidden_dim]
            kv_chunk = kv_chunk.squeeze(1)  # Remove the layer dimension (dimension 1) -> [2, ntokens, hidden_dim]
            kv_chunk = kv_chunk.permute([1, 0, 2])  # Permute to [ntokens, 2, hidden_dim]
            output_fmt = MemoryFormat.KV_T2D
        else:
            # Multi-layer mode: keep KV_2LTD format [2, nlayers, ntokens, hidden_dim]
            output_fmt = MemoryFormat.KV_2LTD

        memory_obj = TensorMemoryObj(
            raw_data=kv_chunk,
            metadata=MemoryObjMetadata(
                shape=kv_chunk.shape,
                dtype=kv_chunk.dtype,
                address=-1,
                phy_size=kv_chunk.numel() * kv_chunk.element_size(),
                ref_count=-1,  # HACK: avoid mis-free
                fmt=output_fmt,
            ),
            parent_allocator=None,
        )

        logger.info(
            f"CacheGenDeserializer.deserialize: RETRIEVED with format={output_fmt}, "
            f"shape={kv_chunk.shape}, dtype={kv_chunk.dtype}, "
            f"layer_id={layer_id}, nlayers={nlayers}"
        )
        return memory_obj
