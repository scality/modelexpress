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
import queue
import re
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import torch

from .obj_transfer import ObjTransferManager, is_obj_available
from .safetensors_meta import SAFETENSORS_DTYPE_MAP, parse_safetensors_header

logger = logging.getLogger("modelexpress.obj_loader")

# Leading "<scheme>://" on an object URI, stripped to get the key prefix.
_URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _configured_load_workers() -> int:
    """Number of shard objects to load concurrently (MX_OBJ_LOAD_WORKERS, default 1).

    Each worker drives its own transfer agent, so this is also the number of
    NIXL agents created. 1 reproduces the classic single-in-flight pipeline.
    A single ObjTransferManager is not thread-safe, so concurrency comes from
    independent managers, one per worker.
    """
    raw = os.environ.get("MX_OBJ_LOAD_WORKERS", "1")
    try:
        n = int(raw)
    except ValueError:
        logger.warning("Invalid MX_OBJ_LOAD_WORKERS=%r; using 1", raw)
        return 1
    return max(1, n)


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
        self._obj_managers: list[ObjTransferManager] = []
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
        # One manager is enough to parse the shard headers below; the full
        # worker pool is sized once the shard count is known.
        self._ensure_obj_managers(1)

        prefix = self._normalize_prefix(object_prefix)
        shard_basenames = self._resolve_shard_basenames(model_ref, revision=revision)

        device = torch.device("cuda", self._device_id)

        shard_jobs = []
        for basename in shard_basenames:
            object_key = self._object_key(prefix, basename)
            tensor_infos = self._parse_object_header(object_key, device)
            if tensor_infos:
                shard_jobs.append((object_key, tensor_infos))

        if not shard_jobs:
            return

        total = len(shard_jobs)
        workers = min(_configured_load_workers(), total)
        self._ensure_obj_managers(workers)

        pbar = None
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(
                total=total,
                desc="Loading safetensors via OBJ",
                unit="shard",
            )

        # Concurrent prefetch: `workers` transfer agents load shards in parallel
        # (agents are not thread-safe, so one manager per worker). Tensors are
        # emitted strictly in shard order via a FIFO window. The window is kept
        # deeper than `workers` (submitted-but-not-yet-emitted futures) so that
        # an out-of-order early finisher never leaves an agent idle behind a
        # straggler: freed agents always have queued work. Memory is bounded to
        # `depth` shards. workers=1 reproduces the classic 1-ahead pipeline.
        depth = min(2 * workers, total)
        mgr_pool: queue.Queue[ObjTransferManager] = queue.Queue()
        for mgr in self._obj_managers[:workers]:
            mgr_pool.put(mgr)

        def _worker(job: tuple[str, dict]) -> dict[str, torch.Tensor]:
            mgr = mgr_pool.get()
            try:
                return self._load_object_tensors(*job, manager=mgr)
            finally:
                mgr_pool.put(mgr)

        pool = ThreadPoolExecutor(max_workers=workers)
        window: deque = deque()
        try:
            next_idx = 0
            while next_idx < total and len(window) < depth:
                window.append(pool.submit(_worker, shard_jobs[next_idx]))
                next_idx += 1

            while window:
                loaded = window.popleft().result()
                if pbar is not None:
                    pbar.update(1)

                if next_idx < total:
                    window.append(pool.submit(_worker, shard_jobs[next_idx]))
                    next_idx += 1

                for name, tensor in loaded.items():
                    yield name, tensor

            logger.info(
                "OBJ load complete in %.2fs (workers=%d)",
                time.perf_counter() - load_start, workers,
            )
        finally:
            if pbar is not None:
                pbar.close()
            pool.shutdown(wait=True)

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

    def _ensure_obj_managers(self, count: int) -> None:
        """Lazily create and initialize up to `count` OBJ transfer managers.

        Each manager is an independent NIXL agent (unique name). Managers are
        created once and reused; growing the pool is idempotent.
        """
        while len(self._obj_managers) < count:
            agent_name = f"mx-obj-{self._device_id}-{uuid.uuid4().hex[:8]}"
            mgr = ObjTransferManager(agent_name=agent_name)
            mgr.initialize()
            self._obj_managers.append(mgr)
        logger.info(
            "OBJ managers ready: %d (device %d)",
            len(self._obj_managers), self._device_id,
        )

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

    def _parse_object_header(
        self, object_key: str, device: torch.device
    ) -> dict[str, dict]:
        """Parse a shard object's safetensors header via ranged RDMA GETs."""
        def read_fn(offset: int, length: int) -> bytes:
            return self._obj_managers[0].read_object_range(
                object_key, offset, length, device
            )

        return parse_safetensors_header(read_fn)

    def _load_object_tensors(
        self,
        object_key: str,
        tensor_infos: dict[str, dict],
        manager: ObjTransferManager | None = None,
    ) -> dict[str, torch.Tensor]:
        """Load all tensors of one shard object via a single OBJ batch transfer.

        `manager` selects the transfer agent (one per concurrent worker);
        defaults to the first pool member for the single-worker path.
        """
        obj_manager = manager if manager is not None else self._obj_managers[0]
        device = torch.device("cuda", self._device_id)

        sorted_names = sorted(
            tensor_infos.keys(),
            key=lambda n: tensor_infos[n]["offset"],
        )

        range_list = []
        tensor_meta = []
        for name in sorted_names:
            info = tensor_infos[name]
            st_dtype = info["dtype"]
            torch_dtype = SAFETENSORS_DTYPE_MAP.get(st_dtype)
            if torch_dtype is None:
                raise RuntimeError(
                    f"Unsupported safetensors dtype '{st_dtype}' for tensor '{name}'"
                )
            range_list.append((info["offset"], info["size"]))
            tensor_meta.append((name, torch_dtype, info["shape"]))

        raw_tensors = obj_manager.batch_load_object(
            object_key, range_list, device,
        )

        result: dict[str, torch.Tensor] = {}
        for raw, (name, torch_dtype, shape) in zip(raw_tensors, tensor_meta, strict=True):
            result[name] = raw.view(torch_dtype).reshape(shape)

        logger.info("Loaded object %s", object_key)
        return result

    def shutdown(self) -> None:
        """Release OBJ resources (all transfer agents in the pool)."""
        for mgr in self._obj_managers:
            mgr.shutdown()
        self._obj_managers = []
