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
    def test_obj_failure_before_any_tensor_is_not_mutated(self, mock_obj_cls):
        """Nothing delivered means the model is untouched, so no rebuild.

        This is the source-unreachable case: the header read 404s and the
        generator raises on its first pull. Reporting mutated here cost a full
        51 GiB model rebuild, which is what ran the GPU out of memory.
        """
        from modelexpress.load_strategy.obj_strategy import ObjStrategy

        def never_yields():
            raise RuntimeError("could not read safetensors headers")
            yield  # pragma: no cover - makes this a generator

        mock_obj = MagicMock()
        mock_obj.load_iter.return_value = never_yields()
        mock_obj_cls.return_value = mock_obj

        ctx = _make_context()
        # A real adapter consumes the iterator; the failure surfaces from it.
        ctx.adapter.apply_weight_iter = MagicMock(
            side_effect=lambda result, it: [_ for _ in it]
        )
        ctx.model_config.model = "test-model"
        ctx.model_config.model_weights = "bucket/prefix"

        with pytest.raises(StrategyFailed, match="safetensors headers") as exc:
            ObjStrategy().load(MagicMock(), ctx)

        assert exc.value.mutated is False, "an untouched model must not be rebuilt"
        mock_obj.shutdown.assert_called_once()

    @patch("modelexpress.obj_loader.MxObjLoader")
    def test_obj_failure_after_some_tensors_is_mutated(self, mock_obj_cls):
        """A load that died part-way through really did write to the model."""
        from modelexpress.load_strategy.obj_strategy import ObjStrategy

        def dies_midway():
            yield ("w0", torch.zeros(1))
            yield ("w1", torch.zeros(1))
            raise RuntimeError("transfer died mid-stream")

        mock_obj = MagicMock()
        mock_obj.load_iter.return_value = dies_midway()
        mock_obj_cls.return_value = mock_obj

        ctx = _make_context()
        ctx.adapter.apply_weight_iter = MagicMock(
            side_effect=lambda result, it: [_ for _ in it]
        )
        ctx.model_config.model = "test-model"
        ctx.model_config.model_weights = "bucket/prefix"

        with pytest.raises(StrategyFailed, match="mid-stream") as exc:
            ObjStrategy().load(MagicMock(), ctx)

        assert exc.value.mutated is True, "a half-written model must be rebuilt"
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


# ---------------------------------------------------------------------------
# Fallback must not carry the failed attempt's frames
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    """A recovered strategy failure leaves no traceback and no retained model."""

    def test_release_failure_frames_clears_the_whole_chain(self):
        from modelexpress.load_strategy import _release_failure_frames

        try:
            try:
                raise ValueError("root")
            except ValueError as root:
                raise RuntimeError("wrapper") from root
        except RuntimeError as e:
            outer = e

        assert outer.__traceback__ is not None
        assert outer.__cause__ is not None
        _release_failure_frames(outer)
        assert outer.__traceback__ is None
        assert outer.__cause__ is None
        assert outer.__context__ is None

    def test_release_failure_frames_survives_a_cycle(self):
        from modelexpress.load_strategy import _release_failure_frames

        a = ValueError("a")
        b = ValueError("b")
        a.__cause__ = b
        b.__cause__ = a  # a chain that points back at itself
        _release_failure_frames(a)  # must terminate
        assert a.__cause__ is None and b.__cause__ is None

    def test_release_model_storage_frees_while_a_reference_is_held(self):
        """The real cause: the caller still names the model in a live frame.

        MxModelLoader.load_model does `model = LoadStrategyChain.run(model, ctx)`, so
        its `model` local points at the original for the whole call. That is not
        garbage and not a cycle, so dropping our references and collecting frees
        nothing -- the old model stays resident while its replacement is built. The
        storage has to be released explicitly. `held` below stands in for the
        caller's frame.
        """
        import torch.nn as tnn
        from modelexpress.engines.vllm.adapter import _release_model_storage

        model = tnn.Linear(64, 64, bias=True)
        model.register_buffer("scratch", torch.zeros(1024))
        held = model  # the caller's live reference; must not prevent the release
        expected = (64 * 64 + 64 + 1024) * 4

        freed = _release_model_storage(model)

        assert freed == expected, f"expected {expected} bytes freed, got {freed}"
        assert held is model, "the module itself is intentionally still reachable"
        assert all(p.numel() == 0 for p in held.parameters())
        assert all(b.numel() == 0 for b in held.buffers())

    def test_release_model_storage_tolerates_none_and_repeats(self):
        from modelexpress.engines.vllm.adapter import _release_model_storage
        import torch.nn as tnn

        assert _release_model_storage(None) == 0
        model = tnn.Linear(8, 8, bias=False)
        assert _release_model_storage(model) == 8 * 8 * 4
        # Idempotent: a second pass finds nothing left and must not double-count.
        assert _release_model_storage(model) == 0
