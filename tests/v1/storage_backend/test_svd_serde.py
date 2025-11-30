# SPDX-License-Identifier: Apache-2.0
"""
Tests for V1 SVD serialization/deserialization.

Tests the full integration path with MemoryObj and layerwise format.
"""
# Third Party
import pytest
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.storage_backend.naive_serde.svd_decoder import SVDDeserializer
from lmcache.v1.storage_backend.naive_serde.svd_encoder import SVDSerializer


@pytest.mark.parametrize("chunk_size", [16, 128, 256])
@pytest.mark.parametrize("rank", [512, 1024])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SVD requires CUDA",
)
def test_v1_svd_layerwise_roundtrip(chunk_size, rank):
    """Test V1 SVD encoder/decoder with layerwise format (KV_T2D)."""
    fmt = "vllm"
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        extra_config={"svd_rank": rank}
    )
    
    metadata = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=torch.bfloat16,
        kv_shape=(8, 128),  # num_heads=8, head_size=128
    )
    
    serializer = SVDSerializer(config, metadata)
    deserializer = SVDDeserializer(config, metadata)
    
    # Generate layerwise KV cache: [num_tokens, 2, hidden_size] (KV_T2D format)
    num_tokens = chunk_size
    hidden_size = 8 * 128  # num_heads * head_size
    kv_layerwise = torch.rand(num_tokens, 2, hidden_size, dtype=torch.bfloat16, device="cuda")
    
    # Create TensorMemoryObj
    memory_obj = TensorMemoryObj(kv_layerwise, dtype=torch.bfloat16, fmt=fmt)
    
    # Serialize
    compressed_obj = serializer.serialize(memory_obj, layer_id=0)
    
    # Deserialize
    reconstructed_obj = deserializer.deserialize(compressed_obj, layer_id=0)
    reconstructed_tensor = reconstructed_obj.tensor
    
    # Check shape matches
    assert reconstructed_tensor.shape == kv_layerwise.shape, \
        f"Shape mismatch: {reconstructed_tensor.shape} vs {kv_layerwise.shape}"
    
    # Check reconstruction quality
    relative_error = torch.norm(kv_layerwise - reconstructed_tensor) / torch.norm(kv_layerwise)
    
    if rank >= 1024:
        assert relative_error < 0.1, f"Relative error too high: {relative_error}"
    else:
        assert relative_error < 0.2, f"Relative error too high: {relative_error}"
    
    # Sanity check
    assert reconstructed_tensor.mean() != 0


@pytest.mark.parametrize("chunk_size", [128])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SVD requires CUDA",
)
def test_v1_svd_compression_ratio(chunk_size):
    """Test that SVD actually compresses data."""
    fmt = "vllm"
    rank = 512  # Half of hidden_dim (1024)
    
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        extra_config={"svd_rank": rank}
    )
    
    metadata = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=torch.bfloat16,
        kv_shape=(8, 128),
    )
    
    serializer = SVDSerializer(config, metadata)
    
    # Generate layerwise KV cache
    num_tokens = chunk_size
    hidden_size = 8 * 128
    kv_layerwise = torch.rand(num_tokens, 2, hidden_size, dtype=torch.bfloat16, device="cuda")
    
    memory_obj = TensorMemoryObj(kv_layerwise, dtype=torch.bfloat16, fmt=fmt)
    
    # Serialize
    compressed_obj = serializer.serialize(memory_obj, layer_id=0)
    
    # Calculate sizes
    original_size = kv_layerwise.numel() * kv_layerwise.element_size()
    compressed_size = len(compressed_obj.bytes_data)
    compression_ratio = original_size / compressed_size
    
    print(f"Original: {original_size} bytes, Compressed: {compressed_size} bytes, "
          f"Ratio: {compression_ratio:.2f}x")
    
    # With rank=512 (half), we expect some compression
    # U: [2, num_tokens, rank], S: [2, rank], Vt: [2, rank, hidden_dim]
    # But torch.save adds overhead, so compression might be modest
    assert compressed_size < original_size, "SVD should compress the data"


@pytest.mark.parametrize("rank", [256, 512, 1024])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SVD requires CUDA",
)
def test_v1_svd_different_ranks(rank):
    """Test SVD with different rank values."""
    fmt = "vllm"
    chunk_size = 128
    
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        extra_config={"svd_rank": rank}
    )
    
    metadata = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=torch.bfloat16,
        kv_shape=(8, 128),
    )
    
    serializer = SVDSerializer(config, metadata)
    deserializer = SVDDeserializer(config, metadata)
    
    # Generate data
    num_tokens = chunk_size
    hidden_size = 8 * 128
    kv_layerwise = torch.rand(num_tokens, 2, hidden_size, dtype=torch.bfloat16, device="cuda")
    
    memory_obj = TensorMemoryObj(kv_layerwise, dtype=torch.bfloat16, fmt=fmt)
    
    # Roundtrip
    compressed_obj = serializer.serialize(memory_obj, layer_id=0)
    reconstructed_obj = deserializer.deserialize(compressed_obj, layer_id=0)
    reconstructed_tensor = reconstructed_obj.tensor
    
    # Check quality varies with rank
    relative_error = torch.norm(kv_layerwise - reconstructed_tensor) / torch.norm(kv_layerwise)
    
    # Lower rank = higher error (more compression)
    # Higher rank = lower error (less compression)
    if rank == 1024:
        # Full rank, should be near-perfect
        assert relative_error < 0.05, f"Full rank should have very low error: {relative_error}"
    elif rank == 512:
        # Half rank, moderate error
        assert relative_error < 0.15, f"Half rank error too high: {relative_error}"
    else:  # rank == 256
        # Quarter rank, higher error acceptable
        assert relative_error < 0.3, f"Quarter rank error too high: {relative_error}"
