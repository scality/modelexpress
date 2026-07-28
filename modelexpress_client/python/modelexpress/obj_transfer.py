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
    MX_OBJ_TIMEOUT: Transfer timeout in seconds (default: 300)

Request size, concurrency and NIC selection are the backend's concern, not ours:
each byte range is handed down as one descriptor and the OBJ backend cuts it into
requests (``split_size``) and bounds how many run at once (``max_inflight``). We
set neither -- ``MX_OBJ_PARAMS`` is forwarded verbatim, so an unset key keeps the
backend's own default rather than one chosen here.

The one exception is ``dram_rdma``, defaulted to false, and it is not tuning: it
states that our host buffers hold metadata rather than bulk data, which no backend
default can infer. See ``obj_backend_params``.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger("modelexpress.obj_transfer")

NIXL_AVAILABLE = False
NixlAgent = None
NixlAgentConfig = None
NixlThreadSync = None
try:
    from nixl._api import nixl_agent as NixlAgent
    from nixl._api import nixl_agent_config as NixlAgentConfig
    from nixl._api import nixl_thread_sync_t as NixlThreadSync
    NIXL_AVAILABLE = True
except ImportError:
    pass

# cuObject caps a single memory registration just under 4 GiB, and the OBJ backend
# applies that cap to every DRAM/VRAM registration. This is a hard backend limit,
# not a tuning knob: ranges above it are split purely to stay registrable.
#
# The exact figure moved between CUDA releases -- cuobjclient.h defines it as
# 4 GiB in 13.2 and 4 GiB - 64 KiB in 13.3 -- so use the smaller one, which is
# valid against both. An exact 4 GiB registration is rejected on 13.3.
MAX_REG_BYTES = 4 * 1024 * 1024 * 1024 - 64 * 1024
_OBJ_BACKEND = "OBJ"

# Packing alignment for host reads. Only the packing is aligned, not the object
# offsets, so this costs at most 63 bytes per range and keeps each range's write
# off a shared cache line with its neighbour.
_HOST_READ_ALIGN = 64


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment

# Backend parameter keys that must never be logged.
_SENSITIVE_PARAMS = frozenset({"access_key", "secret_key", "session_token"})


def obj_backend_params() -> dict[str, str]:
    """Parse the NIXL OBJ backend parameters from MX_OBJ_PARAMS.

    Returns a (possibly empty) map of string->string suitable for
    ``create_backend("OBJ", params)``. Raises ValueError if MX_OBJ_PARAMS
    is set but is not a JSON object.

    Defaults ``dram_rdma`` to false. That is not transport tuning -- it declares how
    we use host memory, which the backend cannot know: our only DRAM use is reading
    metadata, where RDMA costs one memory registration per range and buys nothing.
    An explicit value in MX_OBJ_PARAMS still wins, which is how the two paths get
    compared. A backend that does not know the key ignores it.
    """
    raw = os.environ.get("MX_OBJ_PARAMS", "")
    params: dict[str, str] = {}
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"MX_OBJ_PARAMS is not valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise ValueError("MX_OBJ_PARAMS must be a JSON object of backend params")
        # The NIXL backend expects all parameter values as strings.
        params = {str(k): str(v) for k, v in parsed.items()}
    params.setdefault("dram_rdma", "false")
    return params


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


@dataclass
class ObjBatchHandle:
    """A posted-but-not-yet-awaited OBJ transfer.

    Produced by ``submit_objects`` or ``submit_pooled``, consumed by
    ``wait_objects`` exactly once.

    ``obj_descs``/``vram_descs`` are the registrations to release on completion,
    and are None for a pooled transfer: the pool is registered once for the whole
    load, so releasing per transfer is exactly the cost pooling exists to avoid.
    ``buffers`` is likewise empty for a pooled transfer, whose destination the
    caller already knows as an offset into the pool.
    """

    buffers: list[list[torch.Tensor]]
    label: str
    descriptors: int
    posted_at: float
    xfer: Any
    obj_descs: Any
    vram_descs: Any


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
        # Monotonic OBJ devId source; see submit_objects for why it must not reset.
        self._next_obj_dev_id = 0
        # Pooled path state: one registered staging buffer plus the object-key
        # table, both established once per load. See open_pool().
        self._pool: torch.Tensor | None = None
        self._pool_reg: Any = None
        self._obj_reg: Any = None
        self._obj_dev_ids: dict[str, int] = {}

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
        #
        # sync_mode is explicit: NIXL's default resolves to THREAD_SYNC_NONE when
        # the listener thread is off, which compiles the agent's locks down to
        # no-ops and leaves the OBJ engine's devId->objKey map unguarded. RW takes
        # the writer lock in registerMem and a reader lock on the prepXfer path.
        config = NixlAgentConfig(
            backends=[],
            sync_mode=NixlThreadSync.NIXL_THREAD_SYNC_RW,
        )
        self._agent = NixlAgent(self._agent_name, config)
        self._agent.create_backend(_OBJ_BACKEND, self._params)

        logger.info(
            "OBJ agent '%s' created on device %d (params=%s)",
            self._agent_name, self._device_id, _redact(self._params),
        )

    def batch_load_object(
        self,
        object_key: str,
        range_list: list[tuple[int, int]],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Load multiple byte ranges of one object in a single batch transfer.

        All ranges are submitted at once and the backend decides how to cut and
        schedule them; one range here may become several requests on the wire.

        Args:
            object_key: Object key relative to the OBJ backend's bucket/endpoint.
            range_list: [(object_offset, size), ...]
            device: Target CUDA device.

        Returns:
            List of uint8 GPU tensors (same order as range_list).
        """
        if self._agent is None:
            raise RuntimeError("OBJ agent not initialized")
        return self.batch_load_objects([(object_key, range_list)], device)[0]

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
        return self.wait_objects(self.submit_objects(jobs, device))

    # ------------------------------------------------------------------
    # Pooled path: register once, transfer many
    # ------------------------------------------------------------------

    @property
    def pool(self) -> torch.Tensor | None:
        """The registered staging pool, or None until open_pool()."""
        return self._pool

    def open_pool(self, pool_bytes: int, device: torch.device) -> None:
        """Allocate one staging buffer and register it once for the whole load.

        Registration cost tracks the number of memory regions far more than the
        bytes in them: on an H100 with four rails, ibv_reg_mr costs ~1.0 ms and
        ibv_dereg_mr ~0.65 ms per MR regardless of size, plus only ~0.008 ms/MiB.
        A 4 GiB pool is therefore 4 MRs and 34 ms, where the same bytes registered
        per transfer were 2328 MRs and 2.4s of a Gemma-3-27B load.

        NIXL keeps registration and transfer descriptors separate, so a transfer may
        name any sub-range of registered memory; the DC token client resolves each
        request against the registration containing it. That is what makes reuse
        possible without re-registering.
        """
        if self._agent is None:
            raise RuntimeError("OBJ agent not initialized")
        if self._pool is not None:
            raise RuntimeError("OBJ staging pool already open")
        if pool_bytes > MAX_REG_BYTES:
            raise RuntimeError(
                f"OBJ staging pool of {pool_bytes} bytes exceeds the backend's "
                f"registration limit of {MAX_REG_BYTES}"
            )

        pool = torch.empty(pool_bytes, dtype=torch.uint8, device=device)
        started = time.perf_counter()
        try:
            self._pool_reg = self._agent.register_memory(
                [(pool.data_ptr(), pool_bytes, self._device_id, "")], "VRAM"
            )
        except Exception:
            # Hold nothing on the failure path: this buffer is a large share of
            # VRAM and the caller's next move is a fallback loader that needs it.
            del pool
            torch.cuda.empty_cache()
            raise
        self._pool = pool
        logger.info(
            "OBJ staging pool: %.1f GiB registered once in %.0fms",
            pool_bytes / (1024 ** 3), 1000 * (time.perf_counter() - started),
        )

    def register_objects(self, object_sizes: dict[str, int]) -> None:
        """Seat the object-key table once, one devId per key.

        The backend resolves a transfer's object key from its remote descriptor's
        devId, so the key must be registered before any transfer naming it. For an
        OBJ segment that registration is only a map insert -- no pinning, no
        network -- so doing every key up front is effectively free and removes the
        per-transfer OBJ registration entirely.
        """
        if self._agent is None:
            raise RuntimeError("OBJ agent not initialized")
        if self._obj_reg is not None:
            raise RuntimeError("OBJ keys already registered")

        regions = []
        for key, size in object_sizes.items():
            dev_id = self._next_obj_dev_id
            self._next_obj_dev_id += 1
            self._obj_dev_ids[key] = dev_id
            # len is unused by the OBJ branch of registerMem, which records only
            # devId -> key; pass the highest byte we will read so it is not a lie.
            regions.append((0, max(1, size), dev_id, key))
        self._obj_reg = self._agent.register_memory(regions, "OBJ")
        logger.info("OBJ keys registered: %d object(s)", len(regions))

    def read_ranges_to_host(
        self, reads: list[tuple[str, int, int]]
    ) -> list[bytes]:
        """Read many small object ranges into host memory, as one transfer.

        Takes ``(object_key, object_offset, size)`` and returns the bytes of each
        read, in the order given.

        For metadata, not for weights. Reading N ranges the way weights are read
        costs one registration per range, and registration cost tracks the number
        of memory regions rather than their size: roughly 1.0 ms to pin a region on
        each rail and 0.65 ms to release it, whatever the length. A 127-shard model's
        safetensors headers cost 1016 memory regions and 1.6s that way, to move a few
        MiB. Packing every range into one buffer makes it one registration.

        The buffer is host memory, so the parsed bytes need no device-to-host copy,
        and with the backend's ``dram_rdma=false`` it is not pinned at all -- the
        ranges arrive as plain HTTP response bodies.

        A range may come back short if it reaches past the end of its object; the
        transfer still succeeds and the tail of that range stays zero. Callers that
        over-read on purpose rely on this.
        """
        if self._agent is None:
            raise RuntimeError("OBJ agent not initialized")
        if not reads:
            return []

        # Zeroed, not empty: a short read leaves its tail untouched, and that tail
        # must not read back as whatever was in the page before.
        offsets = []
        total = 0
        for _key, _obj_offset, size in reads:
            total = _align_up(total, _HOST_READ_ALIGN)
            offsets.append(total)
            total += size
        host = torch.zeros(total, dtype=torch.uint8)
        base = host.data_ptr()

        obj_regions = []
        dram_descs_in = []
        obj_descs_in = []
        for (key, obj_offset, size), pool_offset in zip(reads, offsets, strict=True):
            # A distinct devId per range, from the monotonic counter: the backend
            # resolves an object key by the remote descriptor's devId, so sharing one
            # would collapse every key onto a single map slot.
            dev_id = self._next_obj_dev_id
            self._next_obj_dev_id += 1
            obj_regions.append((0, max(1, obj_offset + size), dev_id, key))
            dram_descs_in.append((base + pool_offset, size, 0))
            obj_descs_in.append((obj_offset, size, dev_id))

        obj_reg = self._agent.register_memory(obj_regions, "OBJ")
        try:
            host_reg = self._agent.register_memory([(base, total, 0, "")], "DRAM")
        except Exception:
            # Outside the block below, so the OBJ keys registered above are not
            # left behind when the buffer's own registration is what failed.
            self._agent.deregister_memory(obj_reg)
            raise
        xfer = None
        try:
            dram_descs = self._agent.get_xfer_descs(dram_descs_in, "DRAM")
            obj_descs = self._agent.get_xfer_descs(obj_descs_in, "OBJ")
            xfer = self._agent.initialize_xfer(
                "READ", dram_descs, obj_descs, self._agent.name
            )
            handle = ObjBatchHandle(
                buffers=[],
                label=f"{len(reads)} host range(s) from '{reads[0][0]}'",
                descriptors=len(reads),
                posted_at=time.perf_counter(),
                xfer=xfer,
                obj_descs=None,
                vram_descs=None,
            )
            state = self._agent.transfer(xfer)
            if state == "ERR":
                raise RuntimeError(
                    f"OBJ host read failed for {handle.label}"
                )
            xfer = None  # wait_objects owns the handle from here
            self.wait_objects(handle)
            arr = host.numpy()
            return [
                bytes(arr[pool_offset:pool_offset + size])
                for (_key, _obj_offset, size), pool_offset in zip(
                    reads, offsets, strict=True
                )
            ]
        finally:
            if xfer is not None:
                self._agent.release_xfer_handle(xfer)
            self._free_nixl_memory(obj_reg, host_reg)

    def submit_pooled(
        self, object_key: str, obj_offset: int, size: int, pool_offset: int
    ) -> ObjBatchHandle:
        """Post one range into the pool at pool_offset, without registering.

        Both descriptor lists are built directly rather than derived from a
        registration, which is what keeps this off the pinning path.
        """
        if self._agent is None:
            raise RuntimeError("OBJ agent not initialized")
        if self._pool is None:
            raise RuntimeError("OBJ staging pool not open")
        dev_id = self._obj_dev_ids.get(object_key)
        if dev_id is None:
            raise RuntimeError(f"object key '{object_key}' was not registered")

        vram_descs = self._agent.get_xfer_descs(
            [(self._pool.data_ptr() + pool_offset, size, self._device_id)], "VRAM"
        )
        obj_descs = self._agent.get_xfer_descs(
            [(obj_offset, size, dev_id)], "OBJ"
        )
        xfer = self._agent.initialize_xfer(
            "READ", vram_descs, obj_descs, self._agent.name
        )
        state = self._agent.transfer(xfer)
        if state == "ERR":
            self._agent.release_xfer_handle(xfer)
            raise RuntimeError(f"OBJ transfer failed for key '{object_key}'")

        return ObjBatchHandle(
            buffers=[],
            label=f"key '{object_key}' +{obj_offset}",
            descriptors=1,
            posted_at=time.perf_counter(),
            xfer=xfer,
            obj_descs=None,
            vram_descs=None,
        )

    def close_pool(self) -> None:
        """Release the pool registration and the buffer."""
        if self._agent is not None and self._pool_reg is not None:
            self._agent.deregister_memory(self._pool_reg)
        if self._agent is not None and self._obj_reg is not None:
            self._agent.deregister_memory(self._obj_reg)
        self._pool_reg = None
        self._obj_reg = None
        self._pool = None
        self._obj_dev_ids = {}

    def submit_objects(
        self,
        jobs: list[tuple[str, list[tuple[int, int]]]],
        device: torch.device,
    ) -> ObjBatchHandle:
        """Register and post a multi-object batch without waiting for it.

        Allocates the destination buffers, registers every range, and posts one
        NIXL transfer, then returns immediately. Pair with ``wait_objects``.
        Submitting a second batch before awaiting the first is supported and is
        how the loader keeps the fabric busy across window boundaries.

        Args:
            jobs: list of (object_key, [(object_offset, size), ...]).
            device: Target CUDA device.
        """
        if self._agent is None:
            raise RuntimeError("OBJ agent not initialized")

        result_buffers: list[list[torch.Tensor]] = []
        obj_regions = []
        vram_regions = []

        # The OBJ backend resolves each descriptor's object key by its devId
        # (registerMem stores devIdToObjKey_[devId]=key; prepXfer looks the key
        # up by the remote descriptor's devId). Every object must therefore get a
        # DISTINCT devId -- reusing one collides every key onto a single map slot
        # (last-write-wins, plus a double-free on teardown).
        #
        # The devId comes from a monotonic per-manager counter rather than the
        # index within this batch: with two batches in flight, per-batch indices
        # would alias, and deregistering the older batch would erase the newer
        # batch's map entries. A counter keeps every live registration distinct.
        #
        # Only the OBJ side's devId is free to be an index. On the VRAM side devId
        # must stay the real GPU ordinal: the backend uses it for the CUDA device
        # guard and to pick a PCIe-affine NIC per registration.
        #
        # Each range is one descriptor, whatever its size. The backend cuts it into
        # requests of at most its own split_size and decides which NIC each rides,
        # so request size and rail spread are not ours to choose. The only split
        # here is the cuObject registration ceiling, which is a hard limit rather
        # than a tuning knob.
        for object_key, range_list in jobs:
            obj_dev_id = self._next_obj_dev_id
            self._next_obj_dev_id += 1
            job_bufs: list[torch.Tensor] = []
            for obj_offset, size in range_list:
                buf = torch.empty(size, dtype=torch.uint8, device=device)
                job_bufs.append(buf)
                gpu_base = buf.data_ptr()

                loaded = 0
                while True:
                    span = min(size - loaded, MAX_REG_BYTES)
                    # OBJ descriptor: (offset_in_object, size, devId, object_key).
                    obj_regions.append((obj_offset + loaded, span, obj_dev_id, object_key))
                    vram_regions.append((gpu_base + loaded, span, self._device_id, ""))
                    loaded += span
                    if loaded >= size:
                        break
            result_buffers.append(job_bufs)

        label = (
            f"key '{jobs[0][0]}'" if len(jobs) == 1 else f"{len(jobs)} objects"
        )

        obj_descs = self._agent.register_memory(obj_regions, "OBJ")
        vram_descs = self._agent.register_memory(vram_regions, "VRAM")

        xfer = self._agent.initialize_xfer(
            "READ", vram_descs.trim(), obj_descs.trim(), self._agent.name
        )

        state = self._agent.transfer(xfer)
        if state == "ERR":
            self._agent.release_xfer_handle(xfer)
            self._free_nixl_memory(obj_descs, vram_descs)
            raise RuntimeError(f"OBJ batch transfer failed for {label}")

        return ObjBatchHandle(
            buffers=result_buffers,
            label=label,
            descriptors=len(obj_regions),
            posted_at=time.perf_counter(),
            xfer=xfer,
            obj_descs=obj_descs,
            vram_descs=vram_descs,
        )

    def wait_objects(self, handle: ObjBatchHandle) -> list[list[torch.Tensor]]:
        """Wait for a submitted batch, release its NIXL state, return buffers.

        The timeout runs from the batch's post time, not from entry here, so a
        batch that was queued behind another is not given extra grace.
        """
        if self._agent is None:
            raise RuntimeError("OBJ agent not initialized")

        timeout = float(os.environ.get("MX_OBJ_TIMEOUT", "300"))
        spins = 0
        try:
            while True:
                state = self._agent.check_xfer_state(handle.xfer)
                if state == "DONE":
                    break
                if state == "ERR":
                    raise RuntimeError(
                        f"OBJ batch transfer error for {handle.label}"
                    )
                if time.perf_counter() - handle.posted_at > timeout:
                    raise TimeoutError(
                        f"OBJ batch transfer timeout for {handle.label}"
                    )
                spins += 1
                if spins > 100:
                    time.sleep(0.0001)
                    spins = 0
        finally:
            self._agent.release_xfer_handle(handle.xfer)
            # None for a pooled transfer: the pool stays registered for the whole
            # load, which is the entire point of pooling.
            if handle.obj_descs is not None or handle.vram_descs is not None:
                self._free_nixl_memory(handle.obj_descs, handle.vram_descs)

        return handle.buffers

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

    def _free_nixl_memory(self, obj_descs: Any, local_descs: Any) -> None:
        """Deregister an OBJ registration and its local (VRAM or DRAM) counterpart."""
        self._agent.deregister_memory(obj_descs)
        self._agent.deregister_memory(local_descs)

    def shutdown(self) -> None:
        """Clean up NIXL OBJ resources."""
        # Deregistration needs the agent, so the pool goes first. Idempotent, so
        # a loader that already closed its pool pays nothing.
        self.close_pool()
        self._agent = None
        logger.info("ObjTransferManager shutdown complete")
