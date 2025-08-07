# ruff: noqa: T201

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import nvshmem.core as nvshmem  # type: ignore[import]
import torch
from cuda.core.experimental import Device  # type: ignore[import]
from nvshmem.core import Teams  # type: ignore[import]

from pplx_kernels import PyTorchStreamWrapper, nvshmem_init
from pplx_kernels.all_to_all import AllToAll

from .all_to_all_utils import MoEConfig, RankTestData
from .distributed_utils import (
    ProcessGroupInfo,
    parallel_launch,
    parallel_launch_from_env,
)

logger = logging.getLogger(__name__)


@torch.inference_mode()
def bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    moe: MoEConfig,
) -> tuple[tuple[int, ...], torch.Tensor]:
    device = pgi.device
    num_dp = pgi.world_size // dp_size
    dp_rank = pgi.rank // dp_size

    # Generate the same rank data for each DP group
    rng = torch.Generator()
    rng.manual_seed(dp_rank + 1)
    rank_data = RankTestData(moe, rng, use_max_tokens=True)

    # Allocate symmetric memory
    num_local_experts = moe.num_experts // pgi.world_size

    hidden_dim_scale_bytes = (
        0
        if moe.in_dtype.itemsize != 1
        else (
            (moe.hidden_dim + moe.block_size - 1)
            // moe.block_size
            * torch.float32.itemsize
        )
    )

    ata = AllToAll.internode(
        max_num_tokens=moe.max_num_tokens,
        num_experts=moe.num_experts,
        experts_per_token=moe.experts_per_token,
        rank=pgi.rank,
        world_size=pgi.world_size,
        dp_size=dp_size,
        hidden_dim=moe.hidden_dim,
        hidden_dim_bytes=moe.hidden_dim * moe.in_dtype.itemsize,
        hidden_dim_scale_bytes=hidden_dim_scale_bytes,
    )

    # Allocate input and output tensors
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
                expert_x.size(0),
                expert_x.size(1),
                (expert_x.size(2) + moe.block_size - 1) // moe.block_size,
            ),
            dtype=torch.float32,
            device=device,
        )
    expert_y = expert_x.to(moe.out_dtype)
    dp_x = rank_data.x.to(device)
    dp_x_scale = rank_data.x_scale.to(device) if rank_data.x_scale is not None else None
    indices = rank_data.indices.to(device).to(torch.uint32)
    bound_m = torch.tensor([rank_data.num_tokens], dtype=torch.uint32, device=device)
    weights = rank_data.weights.to(device)
    y = torch.full(
        (moe.max_num_tokens, moe.hidden_dim),
        torch.nan,
        dtype=moe.out_dtype,
        device=device,
    )

    hidden_dim_bytes_with_scale = moe.hidden_dim // dp_size * moe.in_dtype.itemsize
    if moe.in_dtype.itemsize == 1:
        hidden_dim_bytes_with_scale += (
            (moe.hidden_dim // dp_size + moe.block_size - 1)
            // moe.block_size
            * torch.float32.itemsize
        )

    a2a_shape = (
        pgi.world_size,
        num_local_experts,
        moe.max_num_tokens * hidden_dim_bytes_with_scale,
    )
    a2a_tensor = torch.empty(
        a2a_shape,
        dtype=torch.uint8,
        device=device,
    )
    a2a_out_tensor = torch.empty_like(a2a_tensor)

    nvshmem_in = nvshmem.tensor(a2a_shape, dtype=torch.uint8)
    nvshmem_out = nvshmem.tensor(a2a_shape, dtype=torch.uint8)

    # Compute stats
    dispatch_bytes = (
        rank_data.num_tokens * moe.experts_per_token * hidden_dim_bytes_with_scale
    )
    combine_bytes = (
        rank_data.num_tokens
        * moe.experts_per_token
        * (moe.hidden_dim // dp_size)
        * moe.out_dtype.itemsize
    )
    a2a_bytes = a2a_tensor.numel() * a2a_tensor.element_size()
    nvshmem_bytes = nvshmem_in.numel() * nvshmem_in.element_size()

    # Benchmark launcher
    def run() -> tuple[float, ...]:
        num_samples = 10
        events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            for _ in range(num_samples)
        ]

        torch_stream_wrapped = PyTorchStreamWrapper(torch.cuda.current_stream())
        torch_stream_ = torch.cuda.current_stream()

        for e0, e1, e2, e3, e4 in events:
            team = Teams.TEAM_WORLD
            nvshmem.collective.barrier(team, torch_stream_wrapped)

            e0.record(torch_stream_)

            ata.dispatch(
                out_expert_num_tokens=expert_num_tokens,
                out_expert_x=expert_x,
                out_expert_x_scale=expert_x_scale,
                dp_x=dp_x,
                dp_x_scale=dp_x_scale,
                indices=indices,
                bound_m=bound_m,
            )
            e1.record(torch_stream_)

            ata.combine(
                out_tokens=y,
                indices=indices,
                weights=weights,
                expert_y=expert_y,
                bound_m=bound_m,
            )
            e2.record(torch_stream_)

            torch.distributed.all_to_all_single(a2a_out_tensor, a2a_tensor)

            e3.record(torch_stream_)

            nvshmem.collective.alltoall(
                team, nvshmem_out, nvshmem_in, stream=torch_stream_wrapped
            )

            e4.record(torch_stream_)

        # Get latency
        torch_stream_.synchronize()
        sum_dispatch_us = 0.0
        sum_combine_us = 0.0
        sum_a2a_us = 0.0
        sum_nvshmem_us = 0.0
        for e0, e1, e2, e3, e4 in events:
            sum_dispatch_us += e0.elapsed_time(e1) * 1e3
            sum_combine_us += e1.elapsed_time(e2) * 1e3
            sum_a2a_us += e2.elapsed_time(e3) * 1e3
            sum_nvshmem_us += e3.elapsed_time(e4) * 1e3
        dispatch_us = sum_dispatch_us / num_samples
        combine_us = sum_combine_us / num_samples
        a2a_us = sum_a2a_us / num_samples
        nvshmem_us = sum_nvshmem_us / num_samples
        dispatch_gbps = dispatch_bytes / dispatch_us / 1e3
        combine_gbps = combine_bytes / combine_us / 1e3
        a2a_gbps = a2a_bytes / a2a_us / 1e3
        nvshmem_gbps = nvshmem_bytes / nvshmem_us / 1e3
        return (
            dispatch_us,
            combine_us,
            a2a_us,
            nvshmem_us,
            dispatch_gbps,
            combine_gbps,
            a2a_gbps,
            nvshmem_gbps,
        )

    # Warmup
    num_warmup = 10
    with torch.cuda.nvtx.range("warmup"):
        for _ in range(num_warmup):
            run()

    # Benchmark
    torch.distributed.barrier()
    num_repeat = 20
    with torch.cuda.nvtx.range("bench"):
        result = torch.tensor([run() for _ in range(num_repeat)])

    # Cleanup
    ata.destroy()

    nvshmem.free_tensor(nvshmem_in)
    nvshmem.free_tensor(nvshmem_out)

    return (
        (dispatch_bytes, combine_bytes, a2a_bytes, nvshmem_bytes),
        result,
    )


def _worker_bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    in_dtype_str: str,
    out_dtype_str: str,
) -> None:
    num_ranks = pgi.world_size
    global_rank = pgi.rank
    local_rank = pgi.local_rank

    dev = Device(local_rank)
    dev.set_current()

    nvshmem_init(
        global_rank=global_rank, local_rank=local_rank, world_size=num_ranks, device=dev
    )

    in_dtype = getattr(torch, in_dtype_str)
    out_dtype = getattr(torch, out_dtype_str)
    assert isinstance(in_dtype, torch.dtype)
    configs = [
        # V2-Lite:  64 Experts, 6 Experts per Token, 2048 Hidden Dim
        MoEConfig(64, 6, 2048, 1, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 4, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 8, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 16, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 32, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 64, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 128, in_dtype, out_dtype),
        # R1     : 256 Experts, 8 Experts per Token, 7168 Hidden Dim
        MoEConfig(256, 8, 7168, 1, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 4, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 8, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 16, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 32, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 64, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 128, in_dtype, out_dtype),
    ]

    header = [
        "E",
        "E/tok",
        "tok",
        "dim",
        "Dispatch_lat",
        "Dispatch_bw",
        "Dispatch_bytes",
        "Combine_lat",
        "Combine_bw",
        "Combine_bytes",
        "Torch_lat",
        "Torch_bw",
        "Torch_bytes",
        "NVSHMEM_lat",
        "NVSHMEM_bw",
        "NVSHMEM_bytes",
    ]

    outpath = (
        Path(__file__).resolve().parents[1]
        / "data"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_all_to_all.tsv"
    )
    f_out = None
    if pgi.rank == 0:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        f_out = outpath.open("w")

        line = f"EP={pgi.world_size} DP={pgi.world_size // dp_size}"
        print(line)
        f_out.write(line + "\n")

        line = "\t".join(header)
        print(line)
        f_out.write(line + "\n")

    for config in configs:
        if pgi.world_size > config.num_experts:
            continue
        meta, result = bench_all_to_all(pgi, dp_size, config)
        dispatch_bytes, combine_bytes, a2a_bytes, nvshmem_bytes = meta
        if pgi.rank == 0:
            row: dict[str, str] = {
                "E": f"{config.num_experts}",
                "E/tok": f"{config.experts_per_token}",
                "tok": f"{config.max_num_tokens}",
                "dim": f"{config.hidden_dim}",
                "Dispatch_lat": f"{result[:, 0].mean():4.1f}μs ± {result[:, 0].std():4.1f}μs",
                "Dispatch_bw": f"{result[:, 4].mean():2.3f}GB/s",
                "Dispatch_bytes": f"{dispatch_bytes}",
                "Combine_lat": f"{result[:, 1].mean():4.1f}μs ± {result[:, 1].std():4.1f}μs",
                "Combine_bw": f"{result[:, 5].mean():2.3f}GB/s",
                "Combine_bytes": f"{combine_bytes}",
                "Torch_lat": f"{result[:, 2].mean():4.1f}μs ± {result[:, 2].std():4.1f}μs",
                "Torch_bw": f"{result[:, 6].mean():2.3f}GB/s",
                "Torch_bytes": f"{a2a_bytes}",
                "NVSHMEM_lat": f"{result[:, 3].mean():4.1f}μs ± {result[:, 3].std():4.1f}μs",
                "NVSHMEM_bw": f"{result[:, 7].mean():2.3f}GB/s",
                "NVSHMEM_bytes": f"{nvshmem_bytes}",
            }
            assert list(row.keys()) == header
            line = "\t".join(row[h] for h in header)
            print(line)
            assert f_out is not None
            f_out.write(line + "\n")
            f_out.flush()

    if f_out is not None:
        f_out.close()
        print("Saved to", outpath)

    nvshmem.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument(
        "--in-dtype",
        choices=["bfloat16", "float16", "float8_e4m3fn"],
        default="float8_e4m3fn",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    args = parser.parse_args()
    dp_size = int(args.dp_size)
    in_dtype = str(args.in_dtype)
    out_dtype = str(args.out_dtype)

    if "MASTER_ADDR" in os.environ:
        parallel_launch_from_env(_worker_bench_all_to_all, dp_size, in_dtype, out_dtype)
    else:
        world_size = torch.cuda.device_count()
        parallel_launch(
            world_size, _worker_bench_all_to_all, dp_size, in_dtype, out_dtype
        )


if __name__ == "__main__":
    main()
