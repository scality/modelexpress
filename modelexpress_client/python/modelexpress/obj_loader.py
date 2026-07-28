# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Framework-agnostic object-store model loader.

Loads model weights from safetensors objects directly to GPU memory via
NIXL's OBJ plugin, bypassing CPU bounce buffers. The OBJ plugin is
engine-agnostic (plain S3, S3 CRT, or GPU-direct accelerated engines); the
engine and its connection parameters are configured via MX_OBJ_PARAMS and
are opaque to this loader.

The set of shard objects and their names is resolved from the model's
metadata (``model.safetensors.index.json`` / ``model.safetensors``) the same
way the GDS loader resolves local files; only the tiny metadata is read
locally (HF snapshot of JSON files, no weights). Each shard's safetensors
header and tensor data are read directly from the object store via ranged
RDMA GETs, so the bulk weight bytes never touch local disk.

The target GPU is determined from torch.cuda.current_device(), matching the
behavior of vLLM/sglang default loaders.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Iterator, NamedTuple

import torch

from .obj_transfer import (
    MAX_REG_BYTES,
    ObjBatchHandle,
    ObjTransferManager,
    is_obj_available,
)
from .safetensors_meta import (
    HEADER_LEN_SIZE,
    SAFETENSORS_DTYPE_MAP,
    parse_header_json,
    parse_header_size,
)

logger = logging.getLogger("modelexpress.obj_loader")

# Leading "<scheme>://" on an object URI, stripped to get the key prefix.
_URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

# Target bytes per transfer. Tensors are coalesced up to this, which sets the
# descriptor size the backend sees and therefore how it cuts and schedules the
# request stream. Measured optimum for an accelerated engine on 1x100G is a few
# tens of MB per descriptor; it also keeps per-descriptor submission work off the
# critical path on models with very many small tensors (MoE experts, fp8 scales),
# where one transfer per tensor would mean tens of thousands of them.
_DEFAULT_GROUP_BYTES = 64 * 1024 * 1024

# Staging budget: bytes of buffers allowed outstanding, i.e. requested but not yet
# consumed. Its only job is to keep the backend's request queue non-empty; past
# that it buys nothing and takes memory the engine needs.
#
# A flat floor, not a multiple of the largest descriptor. Scaling by the largest
# was wrong because groups are not uniform -- coalescing caps them at the target
# but cannot split a tensor, so one oversized tensor sets the maximum for the whole
# model. Gemma-3-27B's 2.6 GiB embedding produced a 21.0 GiB budget against a mean
# group of 159 MiB, which was 98.6% of the VRAM ceiling and roughly 5x more queue
# than any cap can use.
#
# 4 GiB is what saturates a plausible cap: 512 concurrent requests of 8 MiB. Above
# the cap the extra queue is idle bytes.
#
# Capped by what the backend will register in one call, since the pool is one
# registration: an unreachable default is not a default.
_MIN_STAGING_BYTES = min(4 * 1024 * 1024 * 1024, MAX_REG_BYTES)

# Hard ceiling on the staging budget as a share of currently-free VRAM, so a
# model with very large tensors cannot budget itself into an OOM.
_STAGING_VRAM_CEILING = 0.5


class _PlannedTensor(NamedTuple):
    """One tensor within a group: where it sits in the group's buffer."""

    name: str
    dtype: torch.dtype
    shape: list[int]
    rel_offset: int
    size: int


class _PlannedGroup(NamedTuple):
    """One contiguous byte range to fetch, covering one or more tensors.

    The group is the unit of transfer, so its size is what the backend sees and
    what the staging budget accounts for. The shard survives only as
    ``object_key``; nothing downstream groups by it.
    """

    object_key: str
    offset: int
    size: int
    members: list[_PlannedTensor]


class _PoolRing:
    """Circular allocator over the registered staging pool.

    Groups run from a few KB to several GiB, so fixed-size slots are unusable: slots
    big enough for the largest would give one or two of them and destroy pipelining.
    Allocation is therefore variable-size, at the head, wrapping to 0 when the tail
    of the pool cannot hold the next group.

    Reclaim is guarded by a CUDA event, and that guard is load-bearing. The NIC
    writes into a region outside any CUDA stream, while the consumer's copy out of
    it is enqueued on the current stream. Reusing a region before that copy has
    executed corrupts weights *silently* -- no fault, no error, just wrong numbers.
    So a region stays occupied until an event recorded after its tensors were handed
    over reports complete.
    """

    def __init__(self, size: int):
        self._size = size
        self._head = 0
        # (offset, end, event) in allocation order. event is None while the
        # transfer is still in flight or being consumed.
        self._regions: deque[list] = deque()

    @property
    def size(self) -> int:
        return self._size

    def _free(self, offset: int, size: int) -> bool:
        end = offset + size
        return all(end <= o or offset >= e for o, e, _ in self._regions)

    def _reclaim(self) -> None:
        """Drop regions whose consumer has demonstrably finished with them."""
        while self._regions:
            offset, end, event = self._regions[0]
            if event is None or not event.query():
                return
            self._regions.popleft()

    def alloc(self, size: int) -> int | None:
        """Reserve `size` bytes, or None if only in-flight regions are in the way.

        None means the caller must drain a transfer and retry; everything this can
        resolve on its own -- completed events, and waiting on consumed regions --
        it resolves first.
        """
        if size > self._size:
            raise RuntimeError(
                f"group of {size} bytes exceeds the {self._size}-byte staging pool"
            )
        self._reclaim()
        while True:
            for candidate in (self._head, 0):
                if candidate + size <= self._size and self._free(candidate, size):
                    self._head = candidate + size
                    self._regions.append([candidate, candidate + size, None])
                    return candidate
            # Nothing fits. If the oldest region is merely waiting on its consumer's
            # copy, wait for it rather than reporting failure.
            if self._regions and self._regions[0][2] is not None:
                self._regions[0][2].synchronize()
                self._regions.popleft()
                continue
            return None

    def consumed(self, offset: int) -> None:
        """Mark a region handed to the consumer, recording the reuse barrier.

        Called once control returns from yielding the region's tensors, so the event
        follows every copy the consumer enqueued for them.
        """
        event = torch.cuda.Event()
        event.record()
        for region in self._regions:
            if region[0] == offset:
                region[2] = event
                return


def _element_size(dtype: torch.dtype) -> int:
    """Bytes per element, cached: needed for the view() alignment check."""
    size = _ELEMENT_SIZES.get(dtype)
    if size is None:
        size = torch.empty(0, dtype=dtype).element_size()
        _ELEMENT_SIZES[dtype] = size
    return size


_ELEMENT_SIZES: dict[torch.dtype, int] = {}


class MxObjLoader:
    """
    Load model weights from safetensors objects directly to GPU via the
    NIXL OBJ backend.

    Framework-agnostic. Can be used from vLLM, sglang, or standalone.

    Usage::

        loader = MxObjLoader()
        for name, tensor in loader.load_iter("org/model", "bucket/prefix"):
            process(name, tensor)
    """

    def __init__(self):
        self._obj_manager: ObjTransferManager | None = None
        self._device_id: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_iter(
        self,
        model_ref: str,
        object_prefix: str,
        *,
        use_tqdm: bool = True,
        revision: str | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Yield (tensor_name, gpu_tensor) pairs loaded from object storage.

        Each shard object is batch-loaded through a single OBJ transfer,
        then its tensors are yielded one by one.

        Args:
            model_ref: Model id or local path used to resolve the shard
                layout (index.json / single-file). Only metadata is read.
            object_prefix: Object key prefix where the shard objects live.
        """
        load_start = time.perf_counter()

        if not is_obj_available():
            raise RuntimeError(
                "NIXL OBJ backend is not available. Check libplugin_OBJ.so "
                "and MX_OBJ_ENDPOINT."
            )

        self._device_id = torch.cuda.current_device()
        self._ensure_obj_manager()

        prefix = self._normalize_prefix(object_prefix)
        shard_basenames = self._resolve_shard_basenames(model_ref, revision=revision)

        device = torch.device("cuda", self._device_id)

        plan = self._build_plan(prefix, shard_basenames, device)
        if not plan:
            return

        budget = self._resolve_staging_budget(max(g.size for g in plan))
        ntensors = sum(len(g.members) for g in plan)

        pbar = None
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(
                total=ntensors,
                desc="Loading safetensors via OBJ",
                unit="tensor",
            )

        # One transfer per group into a pool registered once for the whole load.
        #
        # The transfer handle is the unit of completion -- NIXL reports DONE only
        # once every request in a handle has landed, with no per-request status --
        # so the group is also the delivery granularity: its tensors become
        # available together, which is why groups are kept to tens of MB.
        #
        # Concurrency is not managed here. The backend cuts each descriptor into
        # requests and caps how many run at once, so posting generously just keeps
        # its queue fed; what this loop bounds is the pool, and therefore how much
        # VRAM is committed to groups requested but not yet consumed.
        #
        # Entries are popped before being drained, so a failure mid-drain cannot
        # double-await one; whatever is still queued is drained by the finally.
        self._obj_manager.open_pool(budget, device)
        self._obj_manager.register_objects(
            {key: end for key, end in self._object_extents(plan).items()}
        )
        ring = _PoolRing(budget)
        inflight: list[tuple[ObjBatchHandle, _PlannedGroup, int]] = []
        try:
            for group in plan:
                offset = ring.alloc(group.size)
                while offset is None:
                    # Only in-flight regions can be in the way; the ring already
                    # waited on anything merely pending its consumer.
                    if not inflight:
                        raise RuntimeError("staging pool deadlock: nothing to drain")
                    yield from self._drain(inflight.pop(0), ring, pbar)
                    offset = ring.alloc(group.size)

                handle = self._obj_manager.submit_pooled(
                    group.object_key, group.offset, group.size, offset
                )
                inflight.append((handle, group, offset))

            while inflight:
                yield from self._drain(inflight.pop(0), ring, pbar)

            logger.info(
                "OBJ load complete in %.2fs (%d tensors in %d transfers)",
                time.perf_counter() - load_start, ntensors, len(plan),
            )
        finally:
            # A consumer that abandons the iterator (or an error mid-yield) can
            # leave posted transfers unawaited. The server may still be writing into
            # the pool, so they must be awaited before it is released rather than
            # simply dropped.
            for pending, _group, _offset in inflight:
                try:
                    self._obj_manager.wait_objects(pending)
                except Exception as e:
                    logger.warning("Abandoned OBJ transfer failed to drain: %s", e)
            self._obj_manager.close_pool()
            if pbar is not None:
                pbar.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_prefix(object_prefix: str) -> str:
        """Strip an optional URI scheme and trailing slash from the prefix."""
        prefix = _URI_SCHEME_RE.sub("", object_prefix or "")
        return prefix.rstrip("/")

    @staticmethod
    def _object_key(prefix: str, basename: str) -> str:
        """Join the key prefix and a shard basename into an object key."""
        return f"{prefix}/{basename}" if prefix else basename

    def _resolve_staging_budget(self, largest_group: int) -> int:
        """Bytes of staging buffers allowed outstanding at once.

        Sized by what keeps the backend's queue non-empty, not by what VRAM is
        free and not by the largest descriptor. Past saturation extra staging buys
        no throughput and simply takes memory the engine needs.

        Clamped to a share of free VRAM so a model with very large tensors cannot
        budget itself into an OOM, and raised to admit one group when even that is
        too small -- the feed loop always admits one, so the budget should say so.
        """
        override = os.environ.get("MX_OBJ_STAGING_MB")
        free, _total = torch.cuda.mem_get_info(self._device_id)
        # The caching allocator's unused reserve is available to us too.
        cached = torch.cuda.memory_reserved(
            self._device_id
        ) - torch.cuda.memory_allocated(self._device_id)
        available = free + cached

        if override:
            budget = int(override) * 1024 * 1024
            source = "MX_OBJ_STAGING_MB"
        else:
            # Flat: a single oversized tensor must not drag the whole budget up.
            budget = max(_MIN_STAGING_BYTES, largest_group)
            source = "auto"

        ceiling = int(available * _STAGING_VRAM_CEILING)
        if budget > ceiling:
            logger.warning(
                "OBJ staging budget %.1f GiB exceeds %.0f%% of free VRAM "
                "(%.1f GiB free); clamping to %.1f GiB, which may leave the "
                "backend idle between transfers",
                budget / (1024 ** 3), 100 * _STAGING_VRAM_CEILING,
                available / (1024 ** 3), ceiling / (1024 ** 3),
            )
            budget = max(ceiling, largest_group)
            source += ", VRAM-clamped"

        # The pool is one registration, so it cannot exceed what the backend will
        # register. This clamp is last because the two above can raise the budget.
        if budget > MAX_REG_BYTES:
            budget = MAX_REG_BYTES
            source += ", registration-clamped"

        logger.info(
            "OBJ staging budget %.1f GiB (%s); largest descriptor %.1f MiB, "
            "%.1f GiB free",
            budget / (1024 ** 3), source, largest_group / (1024 ** 2),
            available / (1024 ** 3),
        )
        return budget

    def _ensure_obj_manager(self) -> None:
        """Lazily create and initialize the OBJ transfer manager."""
        if self._obj_manager is not None:
            return
        agent_name = f"mx-obj-{self._device_id}-{uuid.uuid4().hex[:8]}"
        self._obj_manager = ObjTransferManager(agent_name=agent_name)
        self._obj_manager.initialize()
        logger.info("OBJ manager initialized for device %d", self._device_id)

    @staticmethod
    def _resolve_metadata_dir(
        model_ref: str, revision: str | None = None
    ) -> str:
        """Resolve a local directory holding the model metadata (no weights)."""
        p = Path(model_ref)
        if p.is_dir():
            return str(p.resolve())

        from huggingface_hub import snapshot_download
        # Only the layout metadata is needed locally; weights stay in the
        # object store and are read over RDMA.
        local_dir = snapshot_download(
            model_ref,
            revision=revision,
            allow_patterns=["*.json", "*.txt", "*.model", "tokenizer*", "*.py"],
        )
        logger.info("Resolved metadata for '%s' -> %s", model_ref, local_dir)
        return local_dir

    def _resolve_shard_basenames(
        self, model_ref: str, revision: str | None = None
    ) -> list[str]:
        """Return the ordered list of shard object basenames for the model."""
        metadata_dir = Path(self._resolve_metadata_dir(model_ref, revision=revision))

        index_path = metadata_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map: dict[str, str] = index.get("weight_map", {})
            if not weight_map:
                raise RuntimeError(f"Empty weight_map in {index_path}")
            basenames = sorted(set(weight_map.values()))
            logger.info(
                "Found sharded model: %d shard objects, %d tensors",
                len(basenames), len(weight_map),
            )
            return basenames

        single = metadata_dir / "model.safetensors"
        if single.exists():
            logger.info("Found single-object model: model.safetensors")
            return ["model.safetensors"]

        raise FileNotFoundError(
            f"No safetensors index or model.safetensors metadata for '{model_ref}'. "
            "Object-store loading requires the shard layout to be resolvable."
        )

    def _parse_object_headers(
        self, object_keys: list[str], device: torch.device
    ) -> dict[str, dict[str, dict]]:
        """Parse every shard's safetensors header in two batched round trips.

        A safetensors header needs two reads -- a u64 length, then that many bytes
        of JSON -- and they are inherently sequential. Done per shard that is
        2 * N sequential round trips before the first weight byte can move, which
        on a many-shard model is the single longest serial stretch of the load.
        Batching across shards makes it 2 regardless of shard count.
        """
        if not object_keys:
            return {}

        sizes = self._obj_manager.batch_load_objects(
            [(key, [(0, HEADER_LEN_SIZE)]) for key in object_keys], device
        )
        header_sizes = [
            parse_header_size(bytes(bufs[0].cpu().numpy())) for bufs in sizes
        ]

        blobs = self._obj_manager.batch_load_objects(
            [
                (key, [(HEADER_LEN_SIZE, size)])
                for key, size in zip(object_keys, header_sizes, strict=True)
            ],
            device,
        )
        return {
            key: parse_header_json(bytes(bufs[0].cpu().numpy()))
            for key, bufs in zip(object_keys, blobs, strict=True)
        }

    def _build_plan(
        self,
        prefix: str,
        shard_basenames: list[str],
        device: torch.device,
    ) -> list[_PlannedGroup]:
        """Flatten every shard's header into an ordered list of groups to fetch.

        Tensors are walked in (shard, offset) order and coalesced into contiguous
        groups of about MX_OBJ_GROUP_MB, so the request stream walks each object
        forwards and one descriptor covers several tensors. Consumption order does
        not have to match: the engine matches tensors by name.

        A group breaks whenever the next tensor would not extend it cleanly:

        - a different object, or a gap in the byte range, since a descriptor is
          one contiguous range of one object;
        - the target size would be exceeded;
        - the tensor would land at a relative offset its dtype cannot be viewed
          at. Slicing the group buffer and calling ``view(dtype)`` needs the
          offset to be a multiple of the element size, and safetensors gives no
          alignment guarantee, so misalignment starts a fresh group rather than
          forcing a copy.

        A tensor is never split across groups, so a tensor larger than the target
        simply gets a group of its own.
        """
        object_keys = [self._object_key(prefix, b) for b in shard_basenames]
        headers = self._parse_object_headers(object_keys, device)

        override = os.environ.get("MX_OBJ_GROUP_MB")
        target = (
            max(1, int(override) * 1024 * 1024) if override else _DEFAULT_GROUP_BYTES
        )

        groups: list[_PlannedGroup] = []
        members: list[_PlannedTensor] = []
        cur_key: str | None = None
        cur_offset = 0
        cur_size = 0
        ntensors = 0

        def flush() -> None:
            nonlocal members, cur_size
            if members:
                groups.append(
                    _PlannedGroup(
                        object_key=cur_key,
                        offset=cur_offset,
                        size=cur_size,
                        members=members,
                    )
                )
                members = []
                cur_size = 0

        for object_key in object_keys:
            tensor_infos = headers[object_key]
            for name in sorted(
                tensor_infos.keys(), key=lambda n: tensor_infos[n]["offset"]
            ):
                info = tensor_infos[name]
                st_dtype = info["dtype"]
                torch_dtype = SAFETENSORS_DTYPE_MAP.get(st_dtype)
                if torch_dtype is None:
                    raise RuntimeError(
                        f"Unsupported safetensors dtype '{st_dtype}' for tensor '{name}'"
                    )
                offset, size = info["offset"], info["size"]
                ntensors += 1

                extends = (
                    members
                    and object_key == cur_key
                    and cur_offset + cur_size == offset
                    and cur_size + size <= target
                    and cur_size % _element_size(torch_dtype) == 0
                )
                if not extends:
                    flush()
                    cur_key, cur_offset = object_key, offset

                members.append(
                    _PlannedTensor(
                        name=name,
                        dtype=torch_dtype,
                        shape=info["shape"],
                        rel_offset=cur_size,
                        size=size,
                    )
                )
                cur_size += size
        flush()

        total = sum(g.size for g in groups)
        logger.info(
            "OBJ plan: %d tensors in %d group(s) across %d shard object(s), "
            "%.1f GiB total, %.1f MiB mean group (target %.0f MiB)",
            ntensors, len(groups), len(shard_basenames), total / (1024 ** 3),
            (total / len(groups) if groups else 0) / (1024 ** 2),
            target / (1024 ** 2),
        )
        return groups

    @staticmethod
    def _object_extents(plan: list[_PlannedGroup]) -> dict[str, int]:
        """Highest byte read from each object, for the one-time key registration."""
        extents: dict[str, int] = {}
        for g in plan:
            end = g.offset + g.size
            if end > extents.get(g.object_key, 0):
                extents[g.object_key] = end
        return extents

    def _drain(
        self,
        inflight: tuple[ObjBatchHandle, _PlannedGroup, int],
        ring: _PoolRing,
        pbar,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Await one posted group, yield its tensors, then free its pool region.

        Members are slices of the pool, so nothing is copied on our side. The region
        is released only after control returns from the last yield, at which point an
        event records the consumer's copies -- see _PoolRing for why reusing it any
        earlier corrupts weights without any visible error.
        """
        handle, group, offset = inflight
        self._obj_manager.wait_objects(handle)
        pool = self._obj_manager.pool

        for m in group.members:
            start = offset + m.rel_offset
            chunk = pool[start:start + m.size]
            yield m.name, chunk.view(m.dtype).reshape(m.shape)

        ring.consumed(offset)
        if pbar is not None:
            pbar.update(len(group.members))

    def shutdown(self) -> None:
        """Release OBJ resources."""
        if self._obj_manager is not None:
            self._obj_manager.shutdown()
            self._obj_manager = None
