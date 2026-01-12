# SPDX-License-Identifier: Apache-2.0
# Standard
import abc
from typing import Optional

# First Party
from lmcache.v1.memory_management import MemoryObj


class Serializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def serialize(
        self, memory_obj: MemoryObj, layer_id: Optional[int] = None
    ) -> MemoryObj:
        """
        Serialize/compress the memory object.

        Input:
            memory_obj: the memory object to be serialized/compressed.
            layer_id: Optional layer ID for layerwise mode. If provided, only bins for this layer will be used.

        Returns:
            MemoryObj: the serialized/compressed memory object.
        """
        raise NotImplementedError


class Deserializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def deserialize(
        self, memory_obj: MemoryObj, layer_id: Optional[int] = None
    ) -> MemoryObj:
        """
        Deserialize/decompress the memory object.

        Input:
            memory_obj: the memory object to be deserialized/decompressed.
            layer_id: Optional layer ID for layerwise mode. If provided and the decoded data
                contains multiple layers, only this layer will be extracted.

        Returns:
            MemoryObj: the deserialized/decompressed memory object.
            None: if the memory allocation fails.
        """
        raise NotImplementedError
