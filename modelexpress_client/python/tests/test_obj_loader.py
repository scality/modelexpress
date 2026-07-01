# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OBJ loader and transfer manager."""

import json
import struct
import threading
import time
from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# OBJ availability detection
# ---------------------------------------------------------------------------


class TestIsObjAvailable:
    """Tests for NIXL OBJ backend availability detection."""

    @patch("modelexpress.obj_transfer.NIXL_AVAILABLE", False)
    def test_nixl_not_installed(self):
        from modelexpress.obj_transfer import is_obj_available
        assert is_obj_available() is False

    @patch("modelexpress.obj_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.obj_transfer._obj_plugin_loadable", return_value=False)
    def test_no_plugin(self, _mock_plugin):
        from modelexpress.obj_transfer import is_obj_available
        assert is_obj_available() is False

    @patch("modelexpress.obj_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.obj_transfer._obj_plugin_loadable", return_value=True)
    def test_all_present(self, _mock_plugin):
        from modelexpress.obj_transfer import is_obj_available
        assert is_obj_available() is True


class TestObjBackendParams:
    """Tests for the generic MX_OBJ_PARAMS pass-through."""

    def test_unset_returns_empty(self):
        from modelexpress.obj_transfer import obj_backend_params
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_PARAMS", None)
            assert obj_backend_params() == {}

    def test_parses_and_stringifies(self):
        from modelexpress.obj_transfer import obj_backend_params
        raw = '{"accelerated": true, "type": "vendor_engine", "num_threads": 8}'
        with patch.dict("os.environ", {"MX_OBJ_PARAMS": raw}):
            params = obj_backend_params()
        assert params == {
            "accelerated": "True",
            "type": "vendor_engine",
            "num_threads": "8",
        }

    def test_invalid_json_raises(self):
        from modelexpress.obj_transfer import obj_backend_params
        with patch.dict("os.environ", {"MX_OBJ_PARAMS": "not-json"}):
            with pytest.raises(ValueError, match="not valid JSON"):
                obj_backend_params()

    def test_non_object_raises(self):
        from modelexpress.obj_transfer import obj_backend_params
        with patch.dict("os.environ", {"MX_OBJ_PARAMS": "[1, 2]"}):
            with pytest.raises(ValueError, match="must be a JSON object"):
                obj_backend_params()


class TestObjPluginLoadable:
    """Tests for libplugin_OBJ.so detection."""

    @patch("ctypes.CDLL")
    def test_loadable(self, mock_cdll):
        from modelexpress.obj_transfer import _obj_plugin_loadable
        assert _obj_plugin_loadable() is True
        mock_cdll.assert_called_once_with("libplugin_OBJ.so")

    @patch("ctypes.CDLL", side_effect=OSError("not found"))
    def test_not_loadable(self, _mock_cdll):
        from modelexpress.obj_transfer import _obj_plugin_loadable
        assert _obj_plugin_loadable() is False


# ---------------------------------------------------------------------------
# ObjTransferManager
# ---------------------------------------------------------------------------


class TestObjTransferManager:
    """Tests for the ObjTransferManager class."""

    def test_not_available_raises(self):
        with patch("modelexpress.obj_transfer.NIXL_AVAILABLE", False):
            from modelexpress.obj_transfer import ObjTransferManager
            mgr = ObjTransferManager(agent_name="test", params={})
            with pytest.raises(RuntimeError, match="not available"):
                mgr.initialize()

    def test_batch_load_requires_init(self):
        with patch("modelexpress.obj_transfer.NIXL_AVAILABLE", True):
            from modelexpress.obj_transfer import ObjTransferManager
            mgr = ObjTransferManager(agent_name="test", params={})
            with pytest.raises(RuntimeError, match="not initialized"):
                mgr.batch_load_object("key", [], torch.device("cpu"))

    def test_max_chunk_capped_at_4gib(self):
        from modelexpress.obj_transfer import ObjTransferManager, _CUOBJ_MAX_REG_SIZE
        with patch.dict("os.environ", {"MX_OBJ_MAX_CHUNK_KB": str(8 * 1024 * 1024)}):
            mgr = ObjTransferManager(agent_name="test", params={})
            assert mgr._max_chunk_size == _CUOBJ_MAX_REG_SIZE


# ---------------------------------------------------------------------------
# Safetensors header parsing (shared helper)
# ---------------------------------------------------------------------------


class TestParseSafetensorsHeader:
    """Tests for the shared safetensors header parser."""

    def test_parses_header_from_reader(self):
        from modelexpress.safetensors_meta import parse_safetensors_header

        header = {"weight": {"dtype": "F32", "shape": [4, 2], "data_offsets": [0, 32]}}
        header_bytes = json.dumps(header).encode()
        blob = struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 32

        def read_fn(offset, length):
            return blob[offset:offset + length]

        parsed = parse_safetensors_header(read_fn)
        assert parsed["weight"]["dtype"] == "F32"
        assert parsed["weight"]["shape"] == [4, 2]
        assert parsed["weight"]["size"] == 32
        assert parsed["weight"]["offset"] == 8 + len(header_bytes)


# ---------------------------------------------------------------------------
# Prefix / key normalization
# ---------------------------------------------------------------------------


class TestObjectKeyResolution:
    """Tests for prefix normalization and object key joins."""

    def test_strips_scheme_and_trailing_slash(self):
        from modelexpress.obj_loader import MxObjLoader
        assert MxObjLoader._normalize_prefix("s3://bucket/models/") == "bucket/models"
        assert MxObjLoader._normalize_prefix("obj://b/p") == "b/p"
        assert MxObjLoader._normalize_prefix("gs://b/p/") == "b/p"
        assert MxObjLoader._normalize_prefix("bucket/p") == "bucket/p"

    def test_object_key_join(self):
        from modelexpress.obj_loader import MxObjLoader
        assert MxObjLoader._object_key("b/p", "model.safetensors") == "b/p/model.safetensors"
        assert MxObjLoader._object_key("", "model.safetensors") == "model.safetensors"


# ---------------------------------------------------------------------------
# Shard layout resolution
# ---------------------------------------------------------------------------


class TestResolveShardBasenames:
    """Tests for shard layout resolution from model metadata."""

    def test_sharded_index(self, tmp_path):
        from modelexpress.obj_loader import MxObjLoader

        index = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        loader = MxObjLoader()
        with patch.object(
            MxObjLoader, "_resolve_metadata_dir", return_value=str(tmp_path)
        ):
            basenames = loader._resolve_shard_basenames("org/model")
        assert basenames == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

    def test_single_file(self, tmp_path):
        from modelexpress.obj_loader import MxObjLoader

        (tmp_path / "model.safetensors").write_bytes(b"\x00")
        loader = MxObjLoader()
        with patch.object(
            MxObjLoader, "_resolve_metadata_dir", return_value=str(tmp_path)
        ):
            assert loader._resolve_shard_basenames("org/model") == ["model.safetensors"]

    def test_no_metadata_raises(self, tmp_path):
        from modelexpress.obj_loader import MxObjLoader

        loader = MxObjLoader()
        with patch.object(
            MxObjLoader, "_resolve_metadata_dir", return_value=str(tmp_path)
        ):
            with pytest.raises(FileNotFoundError, match="No safetensors"):
                loader._resolve_shard_basenames("org/model")


# ---------------------------------------------------------------------------
# Concurrent load pipeline (MX_OBJ_LOAD_WORKERS)
# ---------------------------------------------------------------------------


class _FakeManager:
    """Stand-in for ObjTransferManager: tracks lifecycle + concurrent use."""

    def __init__(self, agent_name=None, params=None):
        self.agent_name = agent_name
        self.initialized = False
        self.shut = False
        self.max_in_use = 0
        self._in_use = 0
        self._lock = threading.Lock()

    def initialize(self):
        self.initialized = True

    def shutdown(self):
        self.shut = True

    # Called from worker threads via _load_object_tensors.
    def enter(self):
        with self._lock:
            self._in_use += 1
            self.max_in_use = max(self.max_in_use, self._in_use)

    def leave(self):
        with self._lock:
            self._in_use -= 1


def _build_index(tmp_path, n_shards, per_shard=2):
    """Write an index.json; return (basenames, tensors_by_basename, ordered)."""
    basenames = [f"model-{i:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
    tensors = {b: [f"{b}::t{j}" for j in range(per_shard)] for b in basenames}
    weight_map = {t: b for b in basenames for t in tensors[b]}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    # Expected yield order: shards sorted, tensors in offset order within a shard.
    ordered = [t for b in sorted(basenames) for t in tensors[b]]
    return basenames, tensors, ordered


class TestLoadIterConcurrency:
    """The MX_OBJ_LOAD_WORKERS parallel prefetch: order, pooling, isolation."""

    def _run(self, tmp_path, monkeypatch, workers, n_shards=6):
        from modelexpress import obj_loader
        from modelexpress.obj_loader import MxObjLoader

        _basenames, tensors, ordered = _build_index(tmp_path, n_shards)

        created: list[_FakeManager] = []

        def _make_manager(agent_name=None, params=None):
            m = _FakeManager(agent_name=agent_name)
            created.append(m)
            return m

        global_lock = threading.Lock()
        state = {"cur": 0, "max": 0}

        def fake_parse(self, object_key, device):
            basename = object_key.split("/")[-1]
            return {t: {"offset": j} for j, t in enumerate(tensors[basename])}

        def fake_load(self, object_key, tensor_infos, manager=None):
            assert isinstance(manager, _FakeManager) and manager.initialized
            manager.enter()
            with global_lock:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            try:
                time.sleep(0.02)  # widen the concurrency window
            finally:
                with global_lock:
                    state["cur"] -= 1
                manager.leave()
            names = sorted(tensor_infos, key=lambda n: tensor_infos[n]["offset"])
            return {n: object_key for n in names}

        monkeypatch.setenv("MX_OBJ_LOAD_WORKERS", str(workers))
        with patch.object(obj_loader, "ObjTransferManager", _make_manager), \
             patch.object(obj_loader, "is_obj_available", return_value=True), \
             patch.object(torch.cuda, "current_device", return_value=0), \
             patch.object(MxObjLoader, "_resolve_metadata_dir", return_value=str(tmp_path)), \
             patch.object(MxObjLoader, "_parse_object_header", fake_parse), \
             patch.object(MxObjLoader, "_load_object_tensors", fake_load):
            loader = MxObjLoader()
            got = [name for name, _ in loader.load_iter("org/model", "buck/pfx", use_tqdm=False)]
            loader.shutdown()

        return got, ordered, created, state

    def test_single_worker_regression(self, tmp_path, monkeypatch):
        got, ordered, created, state = self._run(tmp_path, monkeypatch, workers=1)
        assert got == ordered                       # order unchanged
        assert len(created) == 1                     # single agent
        assert state["max"] == 1                     # never more than one in flight
        assert all(m.shut for m in created)          # torn down

    def test_multi_worker_order_and_pool(self, tmp_path, monkeypatch):
        got, ordered, created, state = self._run(tmp_path, monkeypatch, workers=4, n_shards=6)
        assert got == ordered                        # identical order to K=1
        assert len(created) == 4                      # exactly K agents (pool reused)
        assert all(m.initialized and m.shut for m in created)
        assert state["max"] >= 2                       # genuine concurrency
        assert all(m.max_in_use == 1 for m in created)  # no agent shared across threads

    def test_workers_clamped_to_shard_count(self, tmp_path, monkeypatch):
        got, ordered, created, _ = self._run(tmp_path, monkeypatch, workers=32, n_shards=3)
        assert got == ordered
        assert len(created) == 3                      # min(workers, shards)

    def test_worker_error_propagates_and_shuts_down(self, tmp_path, monkeypatch):
        from modelexpress import obj_loader
        from modelexpress.obj_loader import MxObjLoader

        _basenames, tensors, _ordered = _build_index(tmp_path, 4)
        created: list[_FakeManager] = []

        def _make_manager(agent_name=None, params=None):
            m = _FakeManager(agent_name=agent_name)
            created.append(m)
            return m

        def fake_parse(self, object_key, device):
            basename = object_key.split("/")[-1]
            return {t: {"offset": j} for j, t in enumerate(tensors[basename])}

        def fake_load(self, object_key, tensor_infos, manager=None):
            raise RuntimeError("boom")

        monkeypatch.setenv("MX_OBJ_LOAD_WORKERS", "4")
        with patch.object(obj_loader, "ObjTransferManager", _make_manager), \
             patch.object(obj_loader, "is_obj_available", return_value=True), \
             patch.object(torch.cuda, "current_device", return_value=0), \
             patch.object(MxObjLoader, "_resolve_metadata_dir", return_value=str(tmp_path)), \
             patch.object(MxObjLoader, "_parse_object_header", fake_parse), \
             patch.object(MxObjLoader, "_load_object_tensors", fake_load):
            loader = MxObjLoader()
            with pytest.raises(RuntimeError, match="boom"):
                list(loader.load_iter("org/model", "buck/pfx", use_tqdm=False))
            loader.shutdown()

        assert created and all(m.shut for m in created)

    def test_invalid_workers_defaults_to_one(self, monkeypatch):
        from modelexpress.obj_loader import _configured_load_workers

        monkeypatch.setenv("MX_OBJ_LOAD_WORKERS", "not-a-number")
        assert _configured_load_workers() == 1
        monkeypatch.setenv("MX_OBJ_LOAD_WORKERS", "0")
        assert _configured_load_workers() == 1
        monkeypatch.delenv("MX_OBJ_LOAD_WORKERS", raising=False)
        assert _configured_load_workers() == 1
