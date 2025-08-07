import dataclasses
import logging

import nvshmem.core as nvshmem  # type: ignore[import]
import pytest
import torch
import torch.distributed as dist
from cuda.core.experimental import Device  # type: ignore[import]

from pplx_kernels import nvshmem_init
from pplx_kernels.all_to_all import AllToAll

from .all_to_all_utils import MoEConfig, RankTestData
from .distributed_utils import (
    ProcessGroupInfo,
    parallel_launch,
    parallel_launch_from_env,
    require_multi_node,
)

logger = logging.getLogger(__name__)


small_moe = MoEConfig(
    num_experts=8,
    experts_per_token=3,
    hidden_dim=512,
    max_num_tokens=10,
)

medium_moe = MoEConfig(
    num_experts=64,
    experts_per_token=4,
    hidden_dim=1024,
    max_num_tokens=20,
)


def _str_1d_tensor(t: torch.Tensor) -> str:
    sl = [f"{x:7.4f}" for x in t.tolist()]
    if len(sl) > 5:
        sl = sl[:5] + ["..."]
    return "[" + ", ".join(sl) + "]"


def _do_test_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    moe: MoEConfig,
    internode: bool,
    use_compile: bool,
) -> None:
    rank = pgi.rank
    local_rank = pgi.local_rank
    world_size = pgi.world_size
    dp_rank = rank // dp_size
    num_dp = world_size // dp_size
    assert torch.cuda.current_device() == local_rank
    device = pgi.device

    ata: AllToAll
    if internode:
        ata = AllToAll.internode(
            max_num_tokens=moe.max_num_tokens,
            num_experts=moe.num_experts,
            experts_per_token=moe.experts_per_token,
            rank=rank,
            world_size=world_size,
            dp_size=dp_size,
            hidden_dim=moe.hidden_dim,
            hidden_dim_bytes=moe.hidden_dim * moe.in_dtype.itemsize,
            hidden_dim_scale_bytes=(
                0
                if moe.in_dtype.itemsize != 1
                else (
                    (moe.hidden_dim + moe.block_size - 1)
                    // moe.block_size
                    * torch.float32.itemsize
                )
            ),
        )
    else:
        ata = AllToAll.intranode(
            max_num_tokens=moe.max_num_tokens,
            num_experts=moe.num_experts,
            experts_per_token=moe.experts_per_token,
            rank=rank,
            world_size=world_size,
            dp_size=dp_size,
            hidden_dim=moe.hidden_dim,
            hidden_dim_bytes=moe.hidden_dim * moe.in_dtype.itemsize,
            hidden_dim_scale_bytes=(
                0
                if moe.in_dtype.itemsize != 1
                else (
                    (moe.hidden_dim + moe.block_size - 1)
                    // moe.block_size
                    * torch.float32.itemsize
                )
            ),
        )

    # Generate the same test data on all ranks
    rng = torch.Generator()
    rng.manual_seed(123)
    all_rank_data = [
        RankTestData(moe, rng, use_max_tokens=False) for _ in range(num_dp)
    ]
    rank_data = all_rank_data[dp_rank]

    # Collect info by expert
    expert_token_from: list[list[tuple[int, int]]] = [
        [] for _ in range(moe.num_experts)
    ]
    for i_rank, rd in enumerate(all_rank_data):
        for token_idx in range(rd.num_tokens):
            for expert_idx in rd.indices[token_idx]:
                expert_token_from[expert_idx].append((i_rank, token_idx))

    # Print the test data
    if rank == 0:
        logger.debug("Rank Data:")
        for i_rank, rd in enumerate(all_rank_data):
            logger.debug("  DP Rank %d:", i_rank)
            for token_idx in range(rd.num_tokens):
                indices = rd.indices[token_idx].tolist()
                weights = rd.weights[token_idx].tolist()
                logger.debug(
                    "    x[%d] -> %s",
                    token_idx,
                    list(zip(indices, weights, strict=False)),
                )
            for token_idx in range(rd.num_tokens):
                logger.debug("    x[%d]=%s", token_idx, _str_1d_tensor(rd.x[token_idx]))
            if rd.x_scale is not None:
                for token_idx in range(rd.num_tokens):
                    logger.debug(
                        "    x_scale[%d]=%s",
                        token_idx,
                        _str_1d_tensor(rd.x_scale[token_idx]),
                    )
        for expert_idx in range(moe.num_experts):
            logger.debug(
                "  Expert %d: %d tokens, from: %s",
                expert_idx,
                len(expert_token_from[expert_idx]),
                [f"r{r}t{t}" for r, t in expert_token_from[expert_idx]],
            )

    # Dispatch
    num_local_experts = moe.num_experts // world_size
    expert_num_tokens = torch.empty(
        num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    expert_x = torch.empty(
        (num_local_experts, moe.max_num_tokens * num_dp, moe.hidden_dim),
        dtype=moe.in_dtype,
        device=device,
    )
    expert_x_scale: torch.Tensor | None = None
    if moe.in_dtype.itemsize == 1:
        expert_x_scale = torch.empty(
            (
                num_local_experts,
                expert_x.size(1),
                (expert_x.size(2) + moe.block_size - 1) // moe.block_size,
            ),
            dtype=torch.float32,
            device=device,
        )
    bound_m = torch.tensor([rank_data.num_tokens], dtype=torch.uint32, device=device)
    logger.debug("[rank=%d] Dispatch", rank)

    dispatch = torch.compile(ata.dispatch) if use_compile else ata.dispatch

    dispatch(
        out_expert_num_tokens=expert_num_tokens,
        out_expert_x=expert_x,
        out_expert_x_scale=expert_x_scale,
        dp_x=rank_data.x.to(device),
        dp_x_scale=(
            rank_data.x_scale.to(device) if rank_data.x_scale is not None else None
        ),
        indices=rank_data.indices.to(device).to(torch.uint32),
        bound_m=bound_m,
    )

    torch.cuda.synchronize()
    logger.debug("[rank=%d] Dispatch done", rank)

    # Print and verify the output
    for i_rank in range(world_size):
        if world_size > 1:
            torch.distributed.barrier()
        if i_rank != rank:
            continue
        for i_local_expert in range(num_local_experts):
            expert_idx = i_rank * num_local_experts + i_local_expert
            cnt_tokens = int(expert_num_tokens[i_local_expert].item())
            logger.debug(
                "Expert #%d on Rank %d: %d tokens",
                expert_idx,
                rank,
                cnt_tokens,
            )
            assert cnt_tokens == len(expert_token_from[expert_idx])
            cnt_from_dp_rank = [0] * num_dp
            src_tokens = set()
            src_scales = set()
            dst_tokens = set()
            dst_scales = set()
            for i_token in range(cnt_tokens):
                src_dp_rank, src_token_idx = expert_token_from[expert_idx][i_token]
                cnt_from_dp_rank[src_dp_rank] += 1
                dst_x = expert_x[i_local_expert, i_token]
                src_rank_data = all_rank_data[src_dp_rank]
                src_x = src_rank_data.x[src_token_idx]
                logger.debug(
                    "  x[%d] (from DP Rank %d Token %d): %s",
                    i_token,
                    src_dp_rank,
                    src_token_idx,
                    _str_1d_tensor(dst_x.cpu()),
                )
                dst_tokens.add(tuple(dst_x.cpu().tolist()))
                src_tokens.add(tuple(src_x.cpu().tolist()))

                if moe.in_dtype.itemsize == 1:
                    assert expert_x_scale is not None
                    assert src_rank_data.x_scale is not None
                    dst_x_scale = expert_x_scale[i_local_expert, i_token]
                    src_x_scale = src_rank_data.x_scale[src_token_idx]
                    logger.debug(
                        "  x_scale[%d]                   : %s",
                        i_token,
                        _str_1d_tensor(dst_x_scale.cpu()),
                    )
                    src_scales.add(tuple(src_x_scale.cpu().tolist()))
                    dst_scales.add(tuple(dst_x_scale.cpu().tolist()))

            assert src_scales == dst_scales
            assert src_tokens == dst_tokens

    # Pretend to do some computation
    val = 1.5
    expert_y = expert_x.to(moe.out_dtype) * val

    # Combine
    y = torch.full(
        (moe.max_num_tokens, moe.hidden_dim),
        torch.nan,
        dtype=moe.out_dtype,
        device=device,
    )

    logger.debug("[rank=%d] Combine", rank)

    combine = torch.compile(ata.combine) if use_compile else ata.combine

    combine(
        out_tokens=y,
        indices=rank_data.indices.to(device).to(torch.uint32),
        weights=rank_data.weights.to(device),
        expert_y=expert_y,
        bound_m=bound_m,
    )
    torch.cuda.synchronize()
    logger.debug("[rank=%d] Combine done", rank)

    # Destroy.
    ata.destroy()

    # Verify the output
    ref_y = torch.zeros(
        rank_data.num_tokens, moe.hidden_dim, dtype=y.dtype, device=device
    )
    for i_token in range(rank_data.num_tokens):
        for i_expert in range(moe.experts_per_token):
            expert_idx = int(rank_data.indices[i_token, i_expert].item())
            weight = float(rank_data.weights[i_token, i_expert].item())
            ref_y[i_token] += rank_data.x[i_token].to(device).to(y.dtype) * val * weight
    torch.testing.assert_close(y[: rank_data.num_tokens], ref_y)


def _worker_test_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    in_dtype: str,
    out_dtype: str,
    moe_config: MoEConfig,
    internode: bool,
    use_compile: bool = False,
) -> None:
    num_ranks = dist.get_world_size()

    global_rank = pgi.rank
    local_rank = pgi.local_rank

    dev = Device(local_rank)
    dev.set_current()

    nvshmem_init(
        global_rank=global_rank, local_rank=local_rank, world_size=num_ranks, device=dev
    )

    moe_config = dataclasses.replace(
        moe_config,
        in_dtype=getattr(torch, in_dtype),
        out_dtype=getattr(torch, out_dtype),
    )

    test_script_init_status = nvshmem.direct.init_status()
    if test_script_init_status < 2 and local_rank == 0:
        logger.warning(
            "NVSHMEM hostlib initialization incomplete - status: %d (rank: %d, local_rank: %d)",
            test_script_init_status,
            global_rank,
            local_rank,
        )

    _do_test_all_to_all(pgi, dp_size, moe_config, internode, use_compile)

    nvshmem.finalize()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float8_e4m3fn", "float16"])
@pytest.mark.parametrize("out_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("internode", [True, False])
@pytest.mark.parametrize("use_compile", [False, True])
def test_all_to_all_4_gpu(
    in_dtype: str, out_dtype: str, internode: bool, use_compile: bool
) -> None:
    world_size = 4
    dp_size = 2
    parallel_launch(
        world_size,
        _worker_test_all_to_all,
        dp_size,
        in_dtype,
        out_dtype,
        small_moe,
        internode,
        use_compile,
    )


def _worker_test_all_to_all_multi_node(
    pgi: ProcessGroupInfo,
    in_dtype: str,
    out_dtype: str,
) -> None:
    dp_size = 4
    _worker_test_all_to_all(
        pgi,
        dp_size,
        in_dtype,
        out_dtype,
        medium_moe,
        True,
    )


@require_multi_node
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float8_e4m3fn", "float16"])
@pytest.mark.parametrize("out_dtype", ["float16", "bfloat16"])
def test_all_to_all_multi_node(in_dtype: str, out_dtype: str) -> None:
    parallel_launch_from_env(_worker_test_all_to_all_multi_node, in_dtype, out_dtype)
