# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_basics import CacheGenEncoderOutput
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.svd_decoder import SVDDeserializer
from lmcache.storage_backend.serde.svd_encoder import SVDSerializer


def generate_kv_cache(num_tokens, fmt, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = (
        [num_tokens, num_heads, head_size]
        if fmt == "vllm"
        else [num_heads, num_tokens, head_size]
    )
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def to_blob(kv_tuples):
    return torch.stack(
        [torch.stack(inner_tuple, dim=0) for inner_tuple in kv_tuples], dim=0
    )


@pytest.mark.parametrize("chunk_size", [16, 128, 256])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to CacheGenSerializer",
)
def test_cachegen_encoder(chunk_size):
    fmt = "vllm"
    fmt2 = "huggingface"
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    metadata = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    metadata2 = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt2,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    serializer = CacheGenSerializer(config, metadata)
    serializer2 = CacheGenSerializer(config, metadata2)

    kv = to_blob(generate_kv_cache(chunk_size, fmt, "cuda"))
    output = serializer.to_bytes(kv)
    kv2 = kv.permute([0, 1, 3, 2, 4])
    output2 = serializer2.to_bytes(kv2)

    assert abs(len(output) - len(output2)) < 10
    output_dict = CacheGenEncoderOutput.from_bytes(output)
    assert output_dict.num_heads == 8
    assert output_dict.head_size == 128


@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("chunk_size", [16, 128, 256])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to CacheGenSerializer",
)
def test_cachegen_decoder(fmt, chunk_size):
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    metadata = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata, torch.bfloat16)

    kv = to_blob(generate_kv_cache(chunk_size, fmt, "cuda"))
    output = serializer.to_bytes(kv)

    decoded_kv = deserializer.from_bytes(output)
    assert decoded_kv.shape == kv.shape
    assert decoded_kv.mean() != 0


@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to CacheGenSerializer",
)
def test_cachegen_unmatched_size(fmt):
    chunk_size = 256
    fmt = "vllm"
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    metadata = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata, torch.bfloat16)

    kv = to_blob(generate_kv_cache(chunk_size - 20, fmt, "cuda"))
    output = serializer.to_bytes(kv)

    decoded_kv = deserializer.from_bytes(output)
    assert decoded_kv.shape == kv.shape
    assert decoded_kv.mean() != 0


@pytest.mark.parametrize("chunk_size", [16, 128, 256])
@pytest.mark.parametrize("rank", [512, 1024, 2048])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SVD requires CUDA",
)
def test_svd_encoder(chunk_size, rank):
    """Test SVD encoder with different chunk sizes and ranks."""
    fmt = "vllm"
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    config.svd_rank = rank
    metadata = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=torch.bfloat16,
        kv_shape=(8, 128),  # num_heads=8, head_size=128
    )
    
    serializer = SVDSerializer(config, metadata, torch.bfloat16)
    
    # Generate test KV cache using same helper
    kv = to_blob(generate_kv_cache(chunk_size, fmt, "cuda"))
    output = serializer.to_bytes(kv)
    
    # Check that we got compressed bytes
    assert len(output) > 0
    assert isinstance(output, bytes)


@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("chunk_size", [16, 128, 256])
@pytest.mark.parametrize("rank", [512, 1024])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SVD requires CUDA",
)
def test_svd_decoder(fmt, chunk_size, rank):
    """Test SVD encoder -> decoder roundtrip."""
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    config.svd_rank = rank
    metadata = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=torch.bfloat16,
        kv_shape=(8, 128),  # num_heads=8, head_size=128
    )
    
    serializer = SVDSerializer(config, metadata, torch.bfloat16)
    deserializer = SVDDeserializer(config, metadata, torch.bfloat16)
    
    # Generate test KV cache
    kv = to_blob(generate_kv_cache(chunk_size, fmt, "cuda"))
    
    # Encode and decode
    output = serializer.to_bytes(kv)
    decoded_kv = deserializer.from_bytes(output)
    
    # Check shape matches
    assert decoded_kv.shape == kv.shape
    
    # Check reconstruction quality (lossy compression)
    # Calculate relative error
    diff = torch.abs(kv - decoded_kv)
    relative_error = torch.norm(diff) / torch.norm(kv)
    
    # For reasonable rank, error should be small
    if rank >= 1024:
        assert relative_error < 0.1, f"Relative error too high: {relative_error}"
    
    # Sanity check - decoded tensor should have non-zero values
    assert decoded_kv.mean() != 0


@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SVD requires CUDA",
)
def test_svd_unmatched_size(fmt):
    """Test SVD with non-standard chunk sizes."""
    chunk_size = 256
    rank = 1024
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    config.svd_rank = rank
    metadata = LMCacheEngineMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=torch.bfloat16,
        kv_shape=(8, 128),
    )
    
    serializer = SVDSerializer(config, metadata, torch.bfloat16)
    deserializer = SVDDeserializer(config, metadata, torch.bfloat16)
    
    # Test with smaller chunk size than configured
    kv = to_blob(generate_kv_cache(chunk_size - 20, fmt, "cuda"))
    output = serializer.to_bytes(kv)
    
    decoded_kv = deserializer.from_bytes(output)
    assert decoded_kv.shape == kv.shape
    assert decoded_kv.mean() != 0
    
    # Check reconstruction quality
    relative_error = torch.norm(kv - decoded_kv) / torch.norm(kv)
    assert relative_error < 0.2
