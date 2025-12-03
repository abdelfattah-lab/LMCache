# SPDX-License-Identifier: Apache-2.0
# First Party
from typing import List

from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer


class NaiveSerializer(Serializer):
    def __init__(self):
        pass

    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        memory_obj.ref_count_up()
        return memory_obj

    def serialize_batch(self, memory_objs: List[MemoryObj]) -> List[MemoryObj]:
        return [self.serialize(memory_obj) for memory_obj in memory_objs]


class NaiveDeserializer(Deserializer):
    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        return memory_obj
