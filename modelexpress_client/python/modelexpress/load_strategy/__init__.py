# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
load_strategy: prioritized chain of model loading strategies.

Detects the environment and builds an ordered list of eligible loaders.
MxModelLoader iterates the chain until one succeeds.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from modelexpress.tracing import tracer

from ..adapter import StrategyFailed, UnsupportedCapability
from .base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    SourceTransferError,
    publish_source_if_supported,
    register_tensors,
    publish_metadata,
    unpublish_metadata,
)

__all__ = [
    "LoadContext",
    "LoadResult",
    "LoadStrategy",
    "LoadStrategyChain",
    "SourceTransferError",
    "register_tensors",
    "publish_metadata",
    "unpublish_metadata",
]

logger = logging.getLogger("modelexpress.load_strategy")


def _release_failure_frames(exc: BaseException) -> None:
    """Release the frames a failed attempt's traceback is holding.

    Not about log output. A traceback references every frame of the failed call,
    each frame references its locals, and for a weight loader those locals include
    the model and whatever buffers it was part-way through filling. Recovering from
    the failure while that chain is intact means the old model's VRAM is still
    committed when its replacement is allocated -- 51 GiB twice over on
    Gemma-3-27B, which is an out-of-memory crash rather than a fallback.

    The message has already been logged by the caller; what is dropped here is the
    reference graph, not the diagnosis.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        nxt = current.__cause__ or current.__context__
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
        current = nxt


class LoadStrategyChain:
    """Prioritized chain of model loading strategies.

    Detects the environment, builds an ordered list of eligible loaders,
    and runs them until one succeeds.
    """

    @staticmethod
    def run(model: nn.Module, ctx: LoadContext) -> nn.Module:
        """Build the chain and execute strategies until one succeeds.

        Strategies return LoadResult on success. Expected misses raise
        StrategyFailed; mutated failures trigger adapter re-initialization
        before the next strategy runs. Unexpected exceptions are rolled back
        and treated as fallback to preserve the existing chain behavior.

        Returns the (possibly re-initialized) model on success.
        Raises RuntimeError if no strategy succeeds.
        """
        from .rdma_strategy import RdmaStrategy
        from .obj_strategy import ObjStrategy
        from .model_streamer_strategy import ModelStreamerStrategy
        from .gds_strategy import GdsStrategy
        from .default_strategy import DefaultStrategy

        all_strategies: list[LoadStrategy] = [
            RdmaStrategy(),
            ObjStrategy(),
            ModelStreamerStrategy(),
            GdsStrategy(),
            DefaultStrategy(),
        ]
        eligible = [s for s in all_strategies if s.is_available(ctx)]
        logger.info(f"Eligible loaders: {[s.name for s in eligible]}")

        result = LoadResult(value=model, model=model)
        with tracer.start_as_current_span("Load model") as span:
            span.set_attribute("model_name", ctx.identity.model_name)
            span.set_attribute("global_rank", ctx.global_rank)
            span.set_attribute("eligible_strategies", [s.name for s in eligible])

            for strategy in eligible:
                logger.info(f"[Worker {ctx.global_rank}] Trying strategy: {strategy.name}")
                # Recovery runs AFTER the handler, never inside it. While an except
                # block executes, the interpreter holds the exception, whose traceback
                # references every frame of the failed attempt -- including the
                # engine's weight-loading frame, which holds the model. Rebuilding
                # there leaves the old model resident while its replacement is
                # allocated, so a 51 GiB model needs 102 GiB and the fallback dies of
                # OOM instead of recovering. Leaving the handler releases the frames.
                failure: tuple[str, bool] | None = None
                try:
                    result = strategy.load(result, ctx)
                    publish_source_if_supported(result, ctx)
                    span.set_attribute("weight_loading_strategy", strategy.name)
                    return result.value
                except StrategyFailed as e:
                    failure = (str(e), e.mutated)
                    _release_failure_frames(e)
                except Exception as e:
                    # Unexpected strategy errors should be rare. Keep the engine
                    # alive by falling through to the next strategy; expected
                    # fallback paths should use StrategyFailed instead.
                    #
                    # Logged with the traceback, at WARNING, deliberately: reaching
                    # here means a defect rather than a miss, and the fallback would
                    # otherwise turn it into a silently slower load that nobody
                    # investigates. The stack is the only record of where it came
                    # from, since the frames are released immediately below.
                    logger.warning(
                        "[Worker %s] Strategy %s raised an unexpected error",
                        ctx.global_rank, strategy.name, exc_info=True,
                    )
                    # A strategy that failed unexpectedly may have mutated the model
                    # on its way out, and there is no flag to say otherwise, so treat
                    # it as mutated rather than handing the next strategy a model in
                    # an unknown state.
                    failure = (f"unexpected error: {e}", True)
                    _release_failure_frames(e)

                logger.warning(
                    f"[Worker {ctx.global_rank}] Strategy {strategy.name} failed, "
                    f"trying next: {failure[0]}"
                )
                strategy.rollback(ctx)
                if failure[1]:
                    result = LoadStrategyChain._reinit_for_retry(result, ctx, strategy)
                continue

        raise RuntimeError(
            f"[Worker {ctx.global_rank}] No loading strategy succeeded "
            f"for model '{ctx.identity.model_name}'"
        )

    @staticmethod
    def _reinit_for_retry(
        result: LoadResult,
        ctx: LoadContext,
        strategy: LoadStrategy,
    ) -> LoadResult:
        if ctx.adapter is None:
            raise RuntimeError(
                f"[Worker {ctx.global_rank}] Strategy '{strategy.name}' mutated "
                "the model but no adapter can reinitialize it"
            )
        try:
            return ctx.adapter.reinit_for_retry(result)
        except UnsupportedCapability as exc:
            raise RuntimeError(
                f"[Worker {ctx.global_rank}] Strategy '{strategy.name}' mutated "
                "the model but adapter does not support retry reinitialization"
            ) from exc
