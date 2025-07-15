# pyright: reportCallIssue=false

from typing import Any, Optional

import nvshmem.core as nvshmem  # type: ignore[import]
import torch.distributed as dist


###### NVSHMEM ######
def nvshmem_init(
    global_rank: int,
    local_rank: int,
    world_size: int,
    device: Any,
    uid: Optional[Any] = None,
) -> None:
    uniqueid = nvshmem.get_unique_id(empty=True)
    if local_rank == 0:
        uniqueid = nvshmem.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]

    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()

    nvshmem.init(
        device=device,
        uid=broadcast_objects[0],
        rank=global_rank,
        nranks=world_size,
        initializer_method="uid",
    )


# This stream wrapper returns the format required by CUDA Python. This workaround will be removed when nvshmem4py supports Torch stream interoperability.
# For more information see: https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
class PyTorchStreamWrapper:
    def __init__(self, pt_stream: Any) -> None:
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self) -> tuple[int, int]:
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)
