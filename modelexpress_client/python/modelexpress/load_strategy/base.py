# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and shared helpers for loading strategies."""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Iterator

import torch
import torch.nn as nn

from ..nixl_transfer import is_nixl_available
from ..tensor_utils import log_tensor_summary
from ..metadata.publish import publish_metadata_and_ready
from .context import LoadContext, LoadResult

if TYPE_CHECKING:
    from ..nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.load_strategy")


class SourceTransferError(Exception):
    """Raised when a failure is demonstrably from the remote source side.

    Only this exception triggers marking the source STALE. Target-local errors
    (OOM, process_weights_after_loading, warmup) are left as plain exceptions
    so they propagate without poisoning a healthy source.
    """


class LoadStrategy(ABC):
    """Base class for weight-loading strategies.

    Each strategy is fully self-contained for one loading path. Source
    publication is handled by the chain after a strategy succeeds.

    Contract:
      - return LoadResult only after successful loading
      - raise StrategyFailed(mutated=False) for expected fallback paths
      - raise StrategyFailed(mutated=True) after mutating the model
      - reserve unexpected errors for rare defensive fallback in the chain

    ``mutated`` must be reported, not assumed. It is what makes the chain discard
    the model and build a replacement, so claiming it when nothing was written
    costs a second full copy of the model in device memory for no reason. Wrap the
    weight iterator in WeightDelivery and read ``.mutated`` from it; see that class
    for why a hardcoded True was expensive.
    """

    name: str
    requires: ClassVar[tuple] = ()

    def is_available(self, ctx: LoadContext) -> bool:
        """Check environment: is this strategy usable right now?"""
        if not self.requires:
            return True
        if ctx.adapter is None:
            return False
        cls = type(ctx.adapter)
        return all(getattr(cls, m.__name__) is not m for m in self.requires)

    @abstractmethod
    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        """Attempt to load weights and return the updated result.

        Do not return booleans for fallback. Use StrategyFailed so the chain
        can distinguish clean misses from failures that require re-init.
        """

    def rollback(self, ctx: LoadContext) -> None:
        """Clean up strategy-owned state after a failed load attempt.

        This hook must not decide whether the model is dirty. Strategies report
        that through StrategyFailed(mutated=True).
        """
        return None


# ---------------------------------------------------------------------------
# Shared helpers (used by strategy implementations)
# ---------------------------------------------------------------------------


def _init_nixl_manager(
    global_rank: int, device_id: int, role: str, listen_port: int = 0,
) -> NixlTransferManager:
    """Create and initialize a NIXL transfer manager."""
    from ..nixl_transfer import NixlTransferManager

    agent_name = f"mx-{role}-worker{global_rank}-{uuid.uuid4().hex[:8]}"
    logger.debug(f"[Worker {global_rank}] Initializing NIXL manager with agent_name={agent_name}")
    manager = NixlTransferManager(
        agent_name=agent_name,
        device_id=device_id,
        listen_port=listen_port,
    )
    manager.initialize()
    logger.debug(f"[Worker {global_rank}] NIXL manager initialized")
    return manager


def _as_load_result(result_or_model: LoadResult | nn.Module) -> LoadResult:
    if isinstance(result_or_model, LoadResult):
        return result_or_model
    return LoadResult(value=result_or_model, model=result_or_model)


class WeightDelivery:
    """Wraps a weight iterator and records how many tensors reached the engine.

    A strategy that fails has to tell the chain whether it touched the model,
    because saying yes is expensive: the chain discards the model and builds a
    replacement, and on a large model that means two full copies resident at once.
    Hardcoding "yes" cost a 51 GiB rebuild on a run whose weight source 404'd
    before a single tensor was read, and that rebuild is what ran the GPU out of
    memory.

    Counting deliveries answers it from what happened. Nothing delivered means the
    engine's weight loader was never handed a tensor, so the model is still exactly
    what initialize_model produced and the next strategy can use it as-is -- it
    overwrites every parameter anyway.

    Deliberately conservative in the other direction: a tensor pulled and then
    dropped without being copied still counts, because from here we cannot tell.
    Over-reporting costs a rebuild; under-reporting would hand the next strategy a
    half-written model.
    """

    def __init__(self, source: Iterator[tuple[str, torch.Tensor]]):
        self._source = iter(source)
        self.count = 0

    def __iter__(self) -> WeightDelivery:
        return self

    def __next__(self) -> tuple[str, torch.Tensor]:
        item = next(self._source)
        self.count += 1
        return item

    @property
    def mutated(self) -> bool:
        """Whether the model may have been written to."""
        return self.count > 0


def _metadata_publication_configured(ctx: LoadContext) -> bool:
    """Return whether this worker has a metadata path for P2P serving."""
    server_addr = os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS")
    if server_addr:
        return True
    return getattr(ctx.mx_client, "REQUIRES_P2P_METADATA", False) is True


def register_tensors(
    result_or_model: LoadResult | nn.Module,
    ctx: LoadContext,
    *,
    reuse_discovered: bool = False,
) -> None:
    """Collect model tensors and register them with NIXL.

    Failures are logged but do not raise — the worker continues without
    P2P serving capability.

    Args:
        reuse_discovered: If True, skip tensor discovery
            (``ctx.adapter.discover_tensors``) and re-register the
            previously collected set held in ``ctx.tensors``. Use on
            wake / CRIU-restore paths where the weight set is unchanged
            from initial load.

            Rationale: tensor discovery only needs to run once, when the
            model is in its final post-``process_weights_after_loading``
            state. Re-running it after vLLM's warmup / ``torch.compile`` /
            CUDA graph capture is both unnecessary (the weight set is
            frozen) and risky — post-warmup artifacts such as
            ``torch._dynamo.aot_compile.AOTCompiledFunction`` attach to
            ``model`` as plain attributes and reach framework objects
            (llvmlite, ``torch.distributed`` deprecation shims,
            HuggingFace ``_LazyAutoMapping``) that the walker was never
            designed to traverse. P2P targets receive only the weight
            tensors and run their own warmup / compile locally, so
            compile artifacts are never part of what P2P must expose.

            Falls back to discovery if ``ctx.tensors`` is empty so the
            flag is safe even if the caller misuses it.
    """
    if not _metadata_publication_configured(ctx):
        logger.info(
            f"[Worker {ctx.global_rank}] No MX metadata path configured, "
            "skipping NIXL registration"
        )
        return
    if not is_nixl_available():
        logger.warning(f"[Worker {ctx.global_rank}] NIXL not available, skipping registration")
        return
    if ctx.adapter is None:
        raise RuntimeError("NIXL registration requires an engine adapter")

    try:
        if reuse_discovered and ctx.tensors:
            logger.info(
                f"[Worker {ctx.global_rank}] Re-registering "
                f"{len(ctx.tensors)} previously-discovered tensors "
                f"(skipping model walk)"
            )
        else:
            result = _as_load_result(result_or_model)
            if result.model is None:
                logger.info(
                    f"[Worker {ctx.global_rank}] No model available, skipping NIXL registration"
                )
                return

            ctx.tensors = ctx.adapter.discover_tensors(result)
            log_tensor_summary(ctx.tensors, ctx.global_rank, "Registering tensors")

        if ctx.nixl_manager is None:
            base_port = int(os.environ.get("MX_METADATA_PORT", "5555"))
            listen_port = base_port + ctx.device_id
            ctx.nixl_manager = _init_nixl_manager(
                ctx.global_rank, ctx.device_id, "auto", listen_port
            )

        if not ctx.nixl_manager.tensor_descriptors:
            if ctx.vmm_arena is not None:
                logger.debug(
                    f"[Worker {ctx.global_rank}] Registering arena with NIXL "
                    "(single MR via dmabuf)..."
                )
                ctx.nixl_manager.register_arena(ctx.vmm_arena, ctx.tensors)
            else:
                logger.debug(f"[Worker {ctx.global_rank}] Registering tensors with NIXL...")
                ctx.nixl_manager.register_tensors(ctx.tensors)
            logger.debug(f"[Worker {ctx.global_rank}] Tensors registered with NIXL")
    except Exception as e:
        logger.warning(
            f"[Worker {ctx.global_rank}] NIXL registration failed, "
            f"worker will continue without P2P serving: {e}"
        )


def publish_metadata(ctx: LoadContext) -> None:
    """Publish metadata to the MX server. Failures are logged but do not raise."""
    if ctx.nixl_manager is None:
        logger.info(
            f"[Worker {ctx.global_rank}] No NIXL manager, skipping metadata publish"
        )
        return
    # Decentralized backends (k8s-service) have no central server
    # address; their metadata path is entirely peer-to-peer.
    # Only bail on missing MODEL_EXPRESS_URL / MX_SERVER_ADDRESS when the
    # client actually needs a central coordinator. Strict `is True`
    # check so MagicMock's auto-attribute doesn't masquerade as the flag.
    if not _metadata_publication_configured(ctx):
        logger.info(
            f"[Worker {ctx.global_rank}] No MX server configured, skipping metadata publish"
        )
        return
    try:
        publish_metadata_and_ready(
            ctx.mx_client, ctx.nixl_manager, ctx.tensors,
            ctx.worker_rank, ctx.device_id, ctx.identity, ctx.worker_id,
        )
    except Exception as e:
        logger.warning(
            f"[Worker {ctx.global_rank}] Failed to publish metadata, "
            f"worker will continue without P2P serving: {e}"
        )


def publish_source_if_supported(result: LoadResult, ctx: LoadContext) -> None:
    """Best-effort source publication after a successful load."""
    if result.model_for_publish is None:
        return
    publish_metadata(ctx)


def unpublish_metadata(ctx: LoadContext) -> None:
    """Stop heartbeat, stop worker gRPC server, and mark STALE on MX server.

    Call before memory becomes invalid (e.g., VMM unmap during sleep).
    The NIXL agent stays alive — only the P2P serving state is torn down.
    Call publish_metadata() again after memory is valid to re-enter the
    P2P network.
    """
    from ..metadata.publish import _heartbeat_threads, _worker_servers

    hb = _heartbeat_threads.pop(ctx.worker_rank, None)
    if hb is not None:
        try:
            hb.stop()  # also marks STALE on MX server
            logger.info(f"[Worker {ctx.global_rank}] Heartbeat stopped")
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Failed to stop heartbeat cleanly: {e}"
            )

    ws = _worker_servers.pop(ctx.device_id, None)
    if ws is not None:
        try:
            ws.stop()
            logger.info(f"[Worker {ctx.global_rank}] Worker gRPC server stopped")
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Failed to stop worker gRPC server cleanly: {e}"
            )
