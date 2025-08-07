import logging

import nvshmem.core as nvshmem  # type: ignore[import]
import pytest
import torch
import torch.distributed as dist
from cuda.core.experimental import Device  # type: ignore[import]
from nvshmem.core import Teams  # type: ignore[import]

from pplx_kernels import nvshmem_init

from .distributed_utils import (
    ProcessGroupInfo,
    parallel_launch,
    parallel_launch_from_env,
    require_multi_node,
)

logger = logging.getLogger(__name__)


def test_nvshmem_1_gpu() -> None:
    local_rank = 0
    rank_id = 0  # Define rank_id for single GPU test

    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()

    uniqueid = nvshmem.get_unique_id()
    nvshmem.init(device=dev, uid=uniqueid, rank=0, nranks=1, initializer_method="uid")

    # Check host initialization status
    test_script_init_status = nvshmem.direct.init_status()
    if test_script_init_status < 2 and local_rank == 0:
        logger.warning(
            "NVSHMEM hostlib initialization incomplete - status: %d (rank: %d, local_rank: %d)",
            test_script_init_status,
            rank_id,
            local_rank,
        )

    assert nvshmem.my_pe() == 0
    assert nvshmem.n_pes() == 1

    nvshmem.finalize()


def _worker_test_nvshmem_4_gpu(pgi: ProcessGroupInfo) -> None:
    local_rank = dist.get_rank()

    dev = Device(local_rank)
    dev.set_current()

    nvshmem_init(
        global_rank=pgi.rank,
        local_rank=local_rank,
        world_size=pgi.world_size,
        device=dev,
    )

    # Check host initialization status
    test_script_init_status = nvshmem.direct.init_status()
    if test_script_init_status < 2 and local_rank == 0:
        logger.warning(
            "NVSHMEM hostlib initialization incomplete - status: %d (rank: %d, local_rank: %d)",
            test_script_init_status,
            pgi.rank,
            local_rank,
        )

    assert nvshmem.my_pe() == pgi.rank
    assert nvshmem.n_pes() == pgi.world_size

    nvshmem.finalize()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_nvshmem_4_gpu() -> None:
    parallel_launch(4, _worker_test_nvshmem_4_gpu)


def _worker_test_all_to_all(pgi: ProcessGroupInfo) -> None:
    local_rank = dist.get_rank()

    dev = Device(local_rank)
    dev.set_current()

    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()

    nvshmem_init(
        global_rank=rank_id, local_rank=local_rank, world_size=num_ranks, device=dev
    )

    # Check NVSHMEM host initialization status
    test_script_init_status = nvshmem.direct.init_status()
    if test_script_init_status < 2 and local_rank == 0:
        logger.warning(
            "NVSHMEM hostlib initialization incomplete - status: %d (rank: %d, local_rank: %d)",
            test_script_init_status,
            rank_id,
            local_rank,
        )

    # all-to-all test
    try:
        # Allocate a PyTorch tensor backed by NVSHMEM symmetric memory
        t_in = nvshmem.tensor((pgi.world_size,), dtype=torch.int32).fill_(pgi.rank)
        t_out = nvshmem.tensor((pgi.world_size,), dtype=torch.int32)

        team = Teams.TEAM_WORLD
        nvshmem.collective.alltoall(team, t_out, t_in)

        nvshmem.collective.barrier(team)
        torch.cuda.synchronize()

        assert t_out.tolist() == list(range(pgi.world_size))
    finally:
        nvshmem.free_tensor(t_in)
        nvshmem.free_tensor(t_out)
        nvshmem.finalize()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_all_to_all() -> None:
    parallel_launch(4, _worker_test_all_to_all)


@require_multi_node
def test_all_to_all_multi_node() -> None:
    parallel_launch_from_env(_worker_test_all_to_all)
