# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OBJ loader and transfer manager."""

import json
import struct
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
