# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OBJ loading strategy: object-store-to-GPU loading via the NIXL OBJ plugin.

Reads safetensors weights from an object store directly into GPU memory using
the NIXL OBJ plugin (plain S3, S3 CRT, or a GPU-direct accelerated engine,
selected via ``MX_OBJ_PARAMS``). The shard layout is resolved from the model
metadata (``model_config.model``); the object key prefix comes from
``MX_OBJ_URI``.
"""

from __future__ import annotations

import logging
import os

from ..adapter import EngineAdapter, StrategyFailed
from .base import (
    LoadContext,
    LoadStrategy,
    WeightDelivery,
    _as_load_result,
    register_tensors,
)
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_obj")


def _resolve_object_uri(ctx: LoadContext) -> str:
    """Resolve the object key prefix where the model's shards live."""
    value = getattr(ctx.model_config, "model_weights", None)
    if value:
        return value
    return os.environ.get("MX_OBJ_URI", "")


class ObjStrategy(LoadStrategy):
    """Load weights from object storage via the NIXL OBJ backend."""

    name = "obj"
    requires = (EngineAdapter.apply_weight_iter,)

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        from ..obj_transfer import is_obj_available
        if not is_obj_available():
            logger.info(f"[Worker {ctx.global_rank}] OBJ backend not available, skipping")
            return False
        if not _resolve_object_uri(ctx):
            logger.info(f"[Worker {ctx.global_rank}] MX_OBJ_URI not set, skipping OBJ")
            return False
        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        result = _as_load_result(result)
        from ..obj_loader import MxObjLoader

        object_prefix = _resolve_object_uri(ctx)
        logger.info(
            f"[Worker {ctx.global_rank}] Attempting OBJ loading from {object_prefix}..."
        )
        obj_loader = MxObjLoader()
        try:
            try:
                use_tqdm = getattr(ctx.load_config, "use_tqdm_on_load", True)
                revision = getattr(ctx.model_config, "revision", None)
                weights_iter = obj_loader.load_iter(
                    ctx.model_config.model,
                    object_prefix,
                    use_tqdm=use_tqdm,
                    revision=revision,
                )
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] OBJ loading failed, falling through: {e}"
                )
                raise StrategyFailed(str(e), mutated=False) from e

            delivery = WeightDelivery(weights_iter)
            # Post-load processing rewrites parameters (quantisation prep), so
            # entering it counts as mutation even if no tensor was delivered.
            post_load_started = False
            try:
                result = ctx.adapter.apply_weight_iter(result, delivery)
                logger.info(f"[Worker {ctx.global_rank}] OBJ weight loading complete")
                post_load_started = True
                result = ctx.adapter.after_weight_iter_load(result)
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] OBJ loading failed, falling through: {e}"
                )
                raise StrategyFailed(str(e), mutated=delivery.mutated or post_load_started) from e
        finally:
            obj_loader.shutdown()

        register_tensors(result, ctx)
        return result
