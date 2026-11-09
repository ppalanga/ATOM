# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layouts the Mooncake Store offload moves, and the codec for each.

One row is the codec id folded into the Store namespace and the builder the
worker calls. The scheduler and the worker both read it, so a new layout is
one entry rather than a branch in each caller. Codec modules are imported
here; Triton stays lazy inside the DSV4 codec.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.hybrid.m3 import codec as m3_codec

# Bumped when the dense codec's per-chunk byte layout changes.
DENSE_CODEC_ID = "dense-opaque-block-v1"

CodecBuilder = Callable[[Any, Any, int | None, bool], Any]


@dataclass(frozen=True)
class MooncakeLayout:
    """One layout this connector moves."""

    codec_id: Callable[[], str]
    build: CodecBuilder


def _dense_codec_id() -> str:
    return DENSE_CODEC_ID


def _m3_codec_id() -> str:
    return m3_codec.M3_CODEC_ID


def _build_dense(
    kv_caches: Any,
    _transfer_tensors: Any,
    num_blocks: int | None,
    permit_per_request_state: bool,
) -> DenseKVByteCodec:
    return DenseKVByteCodec(
        kv_caches,
        num_blocks=num_blocks,
        permit_per_request_state=permit_per_request_state,
    )


def _build_m3(
    _kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    _permit_per_request_state: bool,
) -> Any:
    return m3_codec.build_m3_page_codec(transfer_tensors, num_blocks)


MOONCAKE_LAYOUTS: dict[str, MooncakeLayout] = {
    "dense": MooncakeLayout(_dense_codec_id, _build_dense),
    "m3": MooncakeLayout(_m3_codec_id, _build_m3),
}


def require_mooncake_layout(name: str) -> MooncakeLayout:
    """The row for `name`, or the refusal the startup check reports."""
    try:
        return MOONCAKE_LAYOUTS[name]
    except KeyError:
        raise ValueError(
            "the Mooncake Store offload moves only the dense KV layout or M3 "
            f"PAGE layout; this model resolves to {name!r}"
        ) from None


def mooncake_codec_id(config: Any) -> str:
    """The codec version folded into this server's Store namespace."""
    layout = require_mooncake_layout(offcfg.select_offload_layout(config))
    return layout.codec_id()


def build_mooncake_codec(
    config: Any,
    kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    *,
    permit_per_request_state: bool,
) -> Any:
    """Build the codec for the layout `config` resolves to."""
    layout = require_mooncake_layout(offcfg.select_offload_layout(config))
    return layout.build(
        kv_caches,
        transfer_tensors,
        num_blocks,
        permit_per_request_state,
    )
