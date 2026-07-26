# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL OBJ Transfer Manager for direct object-store-to-GPU weight loading.

Uses NIXL's OBJ plugin to move object bytes into GPU memory, bypassing CPU
bounce buffers the way the GDS_MT path does for local NVMe. The OBJ plugin
is engine-agnostic: it supports plain AWS S3, the high-performance S3 CRT
client, and vendor GPU-direct accelerated engines. Which engine is used is
decided entirely by the backend parameter map, not by this module.

The backend parameters (``type``, ``accelerated``, ``endpoint_override``,
``bucket``, ``region``, credentials, ``crtMinLimit``, ``num_threads``, ...)
are supplied verbatim via the ``MX_OBJ_PARAMS`` environment variable (a JSON
object) and passed straight to ``create_backend("OBJ", params)``. See the
NIXL OBJ plugin README for the full parameter vocabulary.

Environment variables:
    MX_OBJ_PARAMS: JSON object of NIXL OBJ backend parameters, passed
                   verbatim to the backend, e.g.
                   '{"bucket":"my-models","region":"us-east-1"}' or, for a
                   GPU-direct accelerated engine,
                   '{"accelerated":"true","type":"<engine>",
                     "endpoint_override":"http://host:port"}'.
    MX_OBJ_MAX_CHUNK_KB: Maximum chunk size in KB (default: 131072 = 128 MB)
    MX_OBJ_TIMEOUT: Transfer timeout in seconds (default: 300)
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import time
from typing import Any

import torch

logger = logging.getLogger("modelexpress.obj_transfer")

NIXL_AVAILABLE = False
NixlAgent = None
NixlAgentConfig = None
try:
    from nixl._api import nixl_agent as NixlAgent
    from nixl._api import nixl_agent_config as NixlAgentConfig
    NIXL_AVAILABLE = True
except ImportError:
    pass

# cuObject caps a single memory registration at 4 GiB; keep chunks below it.
_CUOBJ_MAX_REG_SIZE = 4 * 1024 * 1024 * 1024
_DEFAULT_MAX_CHUNK = 128 * 1024 * 1024  # 128 MB
_OBJ_BACKEND = "OBJ"

# Backend parameter keys that must never be logged.
_SENSITIVE_PARAMS = frozenset({"access_key", "secret_key", "session_token"})


def obj_backend_params() -> dict[str, str]:
    """Parse the NIXL OBJ backend parameters from MX_OBJ_PARAMS.

    Returns a (possibly empty) map of string->string suitable for
    ``create_backend("OBJ", params)``. Raises ValueError if MX_OBJ_PARAMS
    is set but is not a JSON object.
    """
    raw = os.environ.get("MX_OBJ_PARAMS", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"MX_OBJ_PARAMS is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("MX_OBJ_PARAMS must be a JSON object of backend params")
    # The NIXL backend expects all parameter values as strings.
    return {str(k): str(v) for k, v in parsed.items()}


def _redact(params: dict[str, str]) -> dict[str, str]:
    """Return params with sensitive values masked, for logging."""
    return {
        k: ("***" if k in _SENSITIVE_PARAMS else v) for k, v in params.items()
    }


def _obj_plugin_loadable() -> bool:
    """Check if the NIXL OBJ plugin shared library can be loaded."""
    try:
        ctypes.CDLL("libplugin_OBJ.so")
        return True
    except OSError:
        return False


def is_obj_available() -> bool:
    """
    Check whether the NIXL OBJ backend is usable on this host.

    NIXL must be installed and the OBJ plugin shared library must be
    loadable. Backend-specific prerequisites (S3 credentials, the
    cuObject/RDMA fabric for accelerated engines, etc.) are validated
    lazily at backend-creation time in initialize(); a failure there
    surfaces as a load-strategy fall-through rather than a crash.
    """
    if not NIXL_AVAILABLE:
        return False
    if not _obj_plugin_loadable():
        logger.debug("OBJ not available: libplugin_OBJ.so not found")
        return False
    return True


class ObjTransferManager:
    """
    Manages a NIXL OBJ backend for object-to-GPU transfers.

    Engine selection (plain S3, S3 CRT, or a GPU-direct accelerated engine)
    is governed entirely by ``params``; this class is engine-agnostic.

    Supports batch loading: all ranges of one object are submitted in a
    single NIXL transfer so the backend drives them in parallel.

    Usage as context manager::

        with ObjTransferManager(agent_name="mx-obj-0") as obj:
            obj.batch_load_object(key, [(offset, size)], device)
    """

    def __init__(self, agent_name: str, params: dict[str, str] | None = None):
        self._agent_name = agent_name
        self._params = params if params is not None else obj_backend_params()
        self._device_id: int | None = None
        self._agent: Any = None
        override = os.environ.get("MX_OBJ_MAX_CHUNK_KB")
        chunk = int(override) * 1024 if override else _DEFAULT_MAX_CHUNK
        self._max_chunk_size = min(chunk, _CUOBJ_MAX_REG_SIZE)

    def __enter__(self) -> ObjTransferManager:
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
        return None

    @property
    def agent_name(self) -> str:
        return self._agent_name

    def initialize(self) -> None:
        """Initialize the NIXL agent and create the OBJ backend."""
        if not NIXL_AVAILABLE:
            raise RuntimeError(
                "NIXL is not available. Install with: pip install nixl[cu12]"
            )
        if self._agent is not None:
            return

        self._device_id = torch.cuda.current_device()

        # Create the agent without auto-initializing UCX; the OBJ backend
        # needs custom params and is created explicitly below.
        config = NixlAgentConfig(backends=[])
        self._agent = NixlAgent(self._agent_name, config)
        self._agent.create_backend(_OBJ_BACKEND, self._params)

        logger.info(
            "OBJ agent '%s' created on device %d (params=%s, max_chunk=%dMB)",
            self._agent_name, self._device_id, _redact(self._params),
            self._max_chunk_size // (1024 * 1024),
        )

    def batch_load_object(
        self,
        object_key: str,
        range_list: list[tuple[int, int]],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Load multiple byte ranges of one object in a single batch transfer.

        All ranges are submitted at once so the backend drives them in
        parallel. Large ranges are split into chunks of max_chunk_size.

        Args:
            object_key: Object key relative to the OBJ backend's bucket/endpoint.
            range_list: [(object_offset, size), ...]
            device: Target CUDA device.

        Returns:
            List of uint8 GPU tensors (same order as range_list).
        """
        if self._agent is None:
            raise RuntimeError("OBJ agent not initialized")

        max_chunk = self._max_chunk_size

        result_buffers = []
        obj_regions = []
        vram_regions = []

        for obj_offset, size in range_list:
            buf = torch.empty(size, dtype=torch.uint8, device=device)
            result_buffers.append(buf)
            gpu_base = buf.data_ptr()

            loaded = 0
            while loaded < size:
                chunk = min(size - loaded, max_chunk)
                # OBJ descriptor: (offset_in_object, size, devId, object_key).
                # The backend uses the addr field as the read offset and the
                # meta-info string as the object key.
                obj_regions.append((obj_offset + loaded, chunk, 0, object_key))
                vram_regions.append((gpu_base + loaded, chunk, self._device_id, ""))
                loaded += chunk

        obj_descs = self._agent.register_memory(obj_regions, "OBJ")
        vram_descs = self._agent.register_memory(vram_regions, "VRAM")

        handle = self._agent.initialize_xfer(
            "READ", vram_descs.trim(), obj_descs.trim(), self._agent.name
        )

        state = self._agent.transfer(handle)
        if state == "ERR":
            self._agent.release_xfer_handle(handle)
            self._free_nixl_memory(obj_descs, vram_descs)
            raise RuntimeError(f"OBJ batch transfer failed for key '{object_key}'")

        timeout = float(os.environ.get("MX_OBJ_TIMEOUT", "300"))
        t0 = time.perf_counter()
        spins = 0
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                break
            if state == "ERR":
                self._agent.release_xfer_handle(handle)
                self._free_nixl_memory(obj_descs, vram_descs)
                raise RuntimeError(
                    f"OBJ batch transfer error for key '{object_key}'"
                )
            if time.perf_counter() - t0 > timeout:
                self._agent.release_xfer_handle(handle)
                self._free_nixl_memory(obj_descs, vram_descs)
                raise TimeoutError(
                    f"OBJ batch transfer timeout for key '{object_key}'"
                )
            spins += 1
            if spins > 100:
                time.sleep(0.0001)
                spins = 0

        self._agent.release_xfer_handle(handle)
        self._free_nixl_memory(obj_descs, vram_descs)

        return result_buffers

    def batch_load_objects(
        self,
        jobs: list[tuple[str, list[tuple[int, int]]]],
        device: torch.device,
    ) -> list[list[torch.Tensor]]:
        """Load byte ranges from MULTIPLE objects in a single batched transfer.

        Generalizes ``batch_load_object`` across a window of shards. All
        ranges of all jobs are registered up-front and submitted as ONE NIXL
        transfer, so a single agent drives every shard's HTTP+RDMA GETs
        concurrently onto the shared curl-multi poller. Each OBJ descriptor
        carries ITS OWN object key (per-descriptor meta), so shards do not
        need separate agents.

        Because every (job, range) buffer is pre-allocated and each chunk
        descriptor points directly into its target buffer's memory, the
        per-job demux is implicit: result_buffers[job_idx][range_idx] is the
        buffer that range's bytes were written into.

        Args:
            jobs: list of (object_key, [(object_offset, size), ...]).
            device: Target CUDA device.

        Returns:
            List (per job) of lists of uint8 GPU tensors, each inner list in
            the same order as that job's range_list.
        """
        if self._agent is None:
            raise RuntimeError("OBJ agent not initialized")

        max_chunk = self._max_chunk_size

        result_buffers: list[list[torch.Tensor]] = []
        obj_regions = []
        vram_regions = []

        # The OBJ backend resolves each descriptor's object key by its devId
        # (registerMem stores devIdToObjKey_[devId]=key; prepXfer looks the key
        # up by the remote descriptor's devId). To span multiple objects in ONE
        # transfer, every object must therefore get a DISTINCT devId -- reusing
        # devId 0 collides every key onto one map slot (last-write-wins + a
        # double-free on teardown). The job index is that per-object devId; all
        # chunks of one object share it. For a single object (W=1) this is
        # devId 0, identical to batch_load_object.
        for job_idx, (object_key, range_list) in enumerate(jobs):
            job_bufs: list[torch.Tensor] = []
            for obj_offset, size in range_list:
                buf = torch.empty(size, dtype=torch.uint8, device=device)
                job_bufs.append(buf)
                gpu_base = buf.data_ptr()

                loaded = 0
                while loaded < size:
                    chunk = min(size - loaded, max_chunk)
                    # OBJ descriptor: (offset_in_object, size, devId, object_key).
                    obj_regions.append((obj_offset + loaded, chunk, job_idx, object_key))
                    vram_regions.append((gpu_base + loaded, chunk, self._device_id, ""))
                    loaded += chunk
            result_buffers.append(job_bufs)

        obj_descs = self._agent.register_memory(obj_regions, "OBJ")
        vram_descs = self._agent.register_memory(vram_regions, "VRAM")

        handle = self._agent.initialize_xfer(
            "READ", vram_descs.trim(), obj_descs.trim(), self._agent.name
        )

        state = self._agent.transfer(handle)
        if state == "ERR":
            self._agent.release_xfer_handle(handle)
            self._free_nixl_memory(obj_descs, vram_descs)
            raise RuntimeError(
                f"OBJ batch transfer failed for {len(jobs)} objects"
            )

        timeout = float(os.environ.get("MX_OBJ_TIMEOUT", "300"))
        t0 = time.perf_counter()
        spins = 0
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                break
            if state == "ERR":
                self._agent.release_xfer_handle(handle)
                self._free_nixl_memory(obj_descs, vram_descs)
                raise RuntimeError(
                    f"OBJ batch transfer error for {len(jobs)} objects"
                )
            if time.perf_counter() - t0 > timeout:
                self._agent.release_xfer_handle(handle)
                self._free_nixl_memory(obj_descs, vram_descs)
                raise TimeoutError(
                    f"OBJ batch transfer timeout for {len(jobs)} objects"
                )
            spins += 1
            if spins > 100:
                time.sleep(0.0001)
                spins = 0

        self._agent.release_xfer_handle(handle)
        self._free_nixl_memory(obj_descs, vram_descs)

        return result_buffers

    def read_object_range(
        self,
        object_key: str,
        offset: int,
        length: int,
        device: torch.device,
    ) -> bytes:
        """Read a single byte range of an object and return it as host bytes.

        Used for small metadata reads (safetensors headers). The range is
        fetched into GPU memory via the same path and copied back to the host.
        """
        buf = self.batch_load_object(object_key, [(offset, length)], device)[0]
        return bytes(buf.cpu().numpy())

    def _free_nixl_memory(self, obj_descs: Any, vram_descs: Any) -> None:
        """Deregister OBJ and VRAM descriptors from the NIXL agent."""
        self._agent.deregister_memory(obj_descs)
        self._agent.deregister_memory(vram_descs)

    def shutdown(self) -> None:
        """Clean up NIXL OBJ resources."""
        self._agent = None
        logger.info("ObjTransferManager shutdown complete")
