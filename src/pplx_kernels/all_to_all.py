# pyright: reportCallIssue=false

import torch

from .ops import _ops


class AllToAll:
    def __init__(
        self,
        max_num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        rank: int,
        world_size: int,
        dp_size: int,
        hidden_dim: int,
        hidden_dim_bytes: int,
        hidden_dim_scale_bytes: int,
    ) -> None:
        assert world_size % dp_size == 0
        assert world_size // dp_size > 1

        self.world_size = world_size
        self.dp_size = dp_size
        self.max_num_tokens = max_num_tokens
        self._has_scales = hidden_dim_scale_bytes > 0

        self._ptr = _ops.all_to_all_create(
            max_num_tokens,
            num_experts,
            experts_per_token,
            rank,
            world_size,
            dp_size,
            hidden_dim,
            hidden_dim_bytes,
            hidden_dim_scale_bytes,
        )
        assert self._ptr != 0

    def __del__(self) -> None:
        self.destroy()

    def dispatch(
        self,
        out_expert_num_tokens: torch.Tensor,
        out_expert_x: torch.Tensor,
        out_expert_x_scale: torch.Tensor | None,
        dp_x: torch.Tensor,
        dp_x_scale: torch.Tensor | None,
        indices: torch.Tensor,
        bound_m: torch.Tensor | None,
        do_send: bool = True,
        do_recv: bool = True,
    ) -> None:
        assert self._ptr is not None

        if self._has_scales:
            assert out_expert_x_scale is not None
            assert dp_x_scale is not None
        else:
            assert out_expert_x_scale is None
            assert dp_x_scale is None

        _ops.all_to_all_dispatch(
            self._ptr,
            out_expert_num_tokens,
            out_expert_x,
            out_expert_x_scale,
            dp_x,
            dp_x_scale,
            indices,
            bound_m,
            do_send,
            do_recv,
        )

    def combine(
        self,
        out_tokens: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        expert_y: torch.Tensor,
        bound_m: torch.Tensor | None,
        do_send: bool = True,
        do_recv: bool = True,
    ) -> None:
        assert self._ptr is not None
        _ops.all_to_all_combine(
            self._ptr,
            out_tokens,
            indices,
            weights,
            expert_y,
            bound_m,
            do_send,
            do_recv,
        )

    def destroy(self) -> None:
        if self._ptr is not None:
            _ops.all_to_all_destroy(self._ptr)
            self._ptr = None
