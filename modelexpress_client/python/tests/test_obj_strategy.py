# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjStrategy loading behavior and chain registration."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress.adapter import EngineAdapter, StrategyFailed
from modelexpress.load_strategy.context import LoadResult


class _FakeAdapter(EngineAdapter):
    def discover_tensors(self, result: LoadResult):
        return {}

    def after_weight_iter_load(self, result: LoadResult):
        return result

    def apply_weight_iter(self, result: LoadResult, weights_iter):
        if result.model is not None:
            result.model.load_weights(weights_iter)
        return result


def _make_context():
    from modelexpress.load_strategy import LoadContext
    return LoadContext(
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        device_id=0,
        identity=MagicMock(),
        mx_client=MagicMock(),
        worker_id="test-worker",
        adapter=_FakeAdapter(),
    )


class TestObjStrategyAvailability:
    """Tests for ObjStrategy.is_available gating."""

    @patch("modelexpress.obj_transfer.is_obj_available", return_value=False)
    def test_backend_unavailable(self, _mock_avail):
        from modelexpress.load_strategy.obj_strategy import ObjStrategy
        ctx = _make_context()
        assert ObjStrategy().is_available(ctx) is False

    @patch("modelexpress.obj_transfer.is_obj_available", return_value=True)
    def test_no_uri(self, _mock_avail):
        from modelexpress.load_strategy.obj_strategy import ObjStrategy
        ctx = _make_context()
        ctx.model_config.model_weights = None
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_URI", None)
            assert ObjStrategy().is_available(ctx) is False

    @patch("modelexpress.obj_transfer.is_obj_available", return_value=True)
    def test_available_with_uri(self, _mock_avail):
        from modelexpress.load_strategy.obj_strategy import ObjStrategy
        ctx = _make_context()
        ctx.model_config.model_weights = "bucket/prefix"
        assert ObjStrategy().is_available(ctx) is True


class TestObjStrategyIntegration:
    """Tests for ObjStrategy loading behavior."""

    @patch("modelexpress.obj_loader.MxObjLoader")
    @patch("modelexpress.load_strategy.base.publish_metadata_and_ready")
    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=False)
    def test_obj_success(self, _mock_nixl, _mock_pub, mock_obj_cls):
        from modelexpress.load_strategy.obj_strategy import ObjStrategy

        mock_obj = MagicMock()
        mock_obj.load_iter.return_value = iter([("w", torch.zeros(1))])
        mock_obj_cls.return_value = mock_obj

        ctx = _make_context()
        ctx.model_config.model = "test-model"
        ctx.model_config.model_weights = "bucket/prefix"

        model = MagicMock()
        result = ObjStrategy().load(model, ctx)

        assert isinstance(result, LoadResult)
        assert result.model is model
        mock_obj.load_iter.assert_called_once()
        model.load_weights.assert_called_once()
        mock_obj.shutdown.assert_called_once()

    @patch("modelexpress.obj_loader.MxObjLoader")
    def test_obj_failure_raises_strategy_failed(self, mock_obj_cls):
        from modelexpress.load_strategy.obj_strategy import ObjStrategy

        mock_obj = MagicMock()
        mock_obj.load_iter.side_effect = RuntimeError("OBJ error")
        mock_obj_cls.return_value = mock_obj

        ctx = _make_context()
        ctx.model_config.model = "test-model"
        ctx.model_config.model_weights = "bucket/prefix"

        with pytest.raises(StrategyFailed, match="OBJ error") as exc:
            ObjStrategy().load(MagicMock(), ctx)

        assert exc.value.mutated is False
        mock_obj.shutdown.assert_called_once()

    @patch("modelexpress.obj_loader.MxObjLoader")
    def test_obj_apply_weight_iter_failure_is_mutated(self, mock_obj_cls):
        from modelexpress.load_strategy.obj_strategy import ObjStrategy

        mock_obj = MagicMock()
        mock_obj.load_iter.return_value = iter([("w", torch.zeros(1))])
        mock_obj_cls.return_value = mock_obj

        ctx = _make_context()
        ctx.adapter.apply_weight_iter = MagicMock(side_effect=RuntimeError("partial load"))
        ctx.model_config.model = "test-model"
        ctx.model_config.model_weights = "bucket/prefix"

        with pytest.raises(StrategyFailed, match="partial load") as exc:
            ObjStrategy().load(MagicMock(), ctx)

        assert exc.value.mutated is True
        mock_obj.shutdown.assert_called_once()


class TestChainRegistration:
    """ObjStrategy must sit before ModelStreamerStrategy in the chain."""

    def test_obj_before_model_streamer(self):
        import inspect
        from modelexpress.load_strategy import LoadStrategyChain

        src = inspect.getsource(LoadStrategyChain.run)
        obj_idx = src.index("ObjStrategy()")
        ms_idx = src.index("ModelStreamerStrategy()")
        rdma_idx = src.index("RdmaStrategy()")
        assert rdma_idx < obj_idx < ms_idx
