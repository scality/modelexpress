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

    def test_params_reach_backend_verbatim(self):
        # Transport tuning belongs to the backend: we must not inject defaults for
        # split_size or max_inflight, so an unset key keeps the backend's own.
        from modelexpress.obj_transfer import ObjTransferManager

        given = {"bucket": "b", "endpoint_override": "http://h:81"}
        mgr = ObjTransferManager(agent_name="test", params=given)
        seen = {}

        class _Agent:
            name = "test"

            def create_backend(self, backend, params):
                seen.update({"backend": backend, "params": dict(params)})

        with patch("modelexpress.obj_transfer.NIXL_AVAILABLE", True), \
             patch("modelexpress.obj_transfer.NixlAgent", return_value=_Agent()), \
             patch("modelexpress.obj_transfer.NixlAgentConfig"), \
             patch("modelexpress.obj_transfer.NixlThreadSync"), \
             patch("torch.cuda.current_device", return_value=0):
            mgr.initialize()

        assert seen["backend"] == "OBJ"
        assert seen["params"] == given
        assert "max_inflight" not in seen["params"]
        assert "split_size" not in seen["params"]


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
# Test doubles for the loader's feed loop
# ---------------------------------------------------------------------------


def _headers(count, tensors_per_shard=4, tensor_size=1024, dtype="U8"):
    """Parsed-header dicts for `count` shards, tensors laid out contiguously."""
    headers = {}
    for shard in range(count):
        key = f"shard-{shard}.safetensors"
        headers[key] = {
            f"s{shard}.t{i}": {
                "offset": 8 + i * tensor_size,
                "size": tensor_size,
                "dtype": dtype,
                "shape": [tensor_size],
            }
            for i in range(tensors_per_shard)
        }
    return headers


class _FakeManager:
    """Stand-in for ObjTransferManager recording submit/wait ordering.

    Records an `events` list of ("submit", handle, ranges) / ("wait", handle) so a
    test can assert not just what was requested but when it was awaited relative
    to other submissions.
    """

    def __init__(self):
        self.events = []
        self._seq = 0
        self._ranges = {}

    def submit_objects(self, jobs, device):
        self._seq += 1
        handle = _FakeHandle(f"h{self._seq}")
        ranges = tuple((key, tuple(rs)) for key, rs in jobs)
        self._ranges[handle.name] = ranges
        self.events.append(("submit", handle.name, ranges))
        return handle

    def wait_objects(self, handle):
        self.events.append(("wait", handle.name))
        # One uint8 buffer per range, matching what the real manager returns.
        return [
            [torch.zeros(size, dtype=torch.uint8) for _off, size in rs]
            for _key, rs in self._ranges[handle.name]
        ]

    def outstanding(self):
        """Handles submitted but not yet awaited."""
        submitted = [e[1] for e in self.events if e[0] == "submit"]
        awaited = {e[1] for e in self.events if e[0] == "wait"}
        return [h for h in submitted if h not in awaited]

    def submitted_ranges(self):
        """[(key, offset, size), ...] in submission order."""
        return [
            (key, off, size)
            for e in self.events
            if e[0] == "submit"
            for key, rs in e[2]
            for off, size in rs
        ]


class _FakeHandle:
    """Minimal ObjBatchHandle stand-in: only `name` and a settable `buffers`."""

    def __init__(self, name):
        self.name = name
        self.buffers = []


def _loader(manager=None):
    from modelexpress.obj_loader import MxObjLoader

    loader = MxObjLoader()
    loader._device_id = 0
    loader._obj_manager = manager if manager is not None else _FakeManager()
    return loader


def _cuda_mem(free=1 << 40):
    """Patch the three torch.cuda memory queries the budget formula reads."""
    return (
        patch("torch.cuda.mem_get_info", return_value=(free, 1 << 40)),
        patch("torch.cuda.memory_reserved", return_value=0),
        patch("torch.cuda.memory_allocated", return_value=0),
    )


def _plan(loader, headers, **env):
    """Run _build_plan over fake headers, with optional env overrides."""
    with patch.object(loader, "_parse_object_headers", return_value=headers), \
         patch.dict("os.environ", env, clear=False):
        import os
        if "MX_OBJ_GROUP_MB" not in env:
            os.environ.pop("MX_OBJ_GROUP_MB", None)
        return loader._build_plan("", list(headers), None)


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------


class TestCoalescing:
    """Contiguous tensors share one descriptor; a group breaks where it must."""

    def test_merges_contiguous_tensors_up_to_target(self):
        loader = _loader()
        # 8 tensors of 1 MiB, 4 MiB target -> 2 groups of 4.
        headers = _headers(1, tensors_per_shard=8, tensor_size=1024 * 1024)
        plan = _plan(loader, headers, MX_OBJ_GROUP_MB="4")

        assert len(plan) == 2
        assert [len(g.members) for g in plan] == [4, 4]
        assert [g.size for g in plan] == [4 * 1024 * 1024] * 2
        # Members carry offsets relative to their group's base.
        assert [m.rel_offset for m in plan[0].members] == [
            0, 1024 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024
        ]

    def test_breaks_between_objects(self):
        loader = _loader()
        # Two shards, tiny tensors, huge target: must still not merge across keys.
        headers = _headers(2, tensors_per_shard=2, tensor_size=16)
        plan = _plan(loader, headers, MX_OBJ_GROUP_MB="64")

        assert len(plan) == 2
        assert {g.object_key for g in plan} == set(headers)
        assert all(len(g.members) == 2 for g in plan)

    def test_breaks_on_gap(self):
        loader = _loader()
        headers = {
            "s.safetensors": {
                "a": {"offset": 8, "size": 16, "dtype": "U8", "shape": [16]},
                # 8-byte hole between a and b.
                "b": {"offset": 32, "size": 16, "dtype": "U8", "shape": [16]},
            }
        }
        plan = _plan(loader, headers, MX_OBJ_GROUP_MB="64")

        assert len(plan) == 2, "a gap must start a new descriptor"
        assert [g.offset for g in plan] == [8, 32]

    def test_breaks_on_dtype_misalignment(self):
        loader = _loader()
        # 3 bytes of U8 then an F32: a slice at relative offset 3 cannot be
        # viewed as float32, so the F32 must start its own group.
        headers = {
            "s.safetensors": {
                "a": {"offset": 8, "size": 3, "dtype": "U8", "shape": [3]},
                "b": {"offset": 11, "size": 8, "dtype": "F32", "shape": [2]},
            }
        }
        plan = _plan(loader, headers, MX_OBJ_GROUP_MB="64")

        assert len(plan) == 2
        assert [m.name for g in plan for m in g.members] == ["a", "b"]
        assert plan[1].members[0].rel_offset == 0

    def test_merges_when_alignment_happens_to_work(self):
        loader = _loader()
        # 4 bytes of U8 then an F32 -> relative offset 4, divisible by 4.
        headers = {
            "s.safetensors": {
                "a": {"offset": 8, "size": 4, "dtype": "U8", "shape": [4]},
                "b": {"offset": 12, "size": 8, "dtype": "F32", "shape": [2]},
            }
        }
        plan = _plan(loader, headers, MX_OBJ_GROUP_MB="64")

        assert len(plan) == 1
        assert [m.rel_offset for m in plan[0].members] == [0, 4]

    def test_oversized_tensor_gets_its_own_group(self):
        loader = _loader()
        headers = _headers(1, tensors_per_shard=3, tensor_size=8 * 1024 * 1024)
        plan = _plan(loader, headers, MX_OBJ_GROUP_MB="1")

        assert len(plan) == 3, "a tensor is never split across descriptors"
        assert all(len(g.members) == 1 for g in plan)
        assert all(g.size == 8 * 1024 * 1024 for g in plan)

    def test_default_target_is_64mib(self):
        from modelexpress.obj_loader import _DEFAULT_GROUP_BYTES

        assert _DEFAULT_GROUP_BYTES == 64 * 1024 * 1024
        loader = _loader()
        # 4 MiB tensors with the default target -> 16 per group.
        headers = _headers(1, tensors_per_shard=32, tensor_size=4 * 1024 * 1024)
        plan = _plan(loader, headers)
        assert [len(g.members) for g in plan] == [16, 16]

    def test_rejects_unknown_dtype(self):
        loader = _loader()
        headers = {
            "s.safetensors": {
                "t": {"offset": 8, "size": 4, "dtype": "NOPE", "shape": [4]}
            }
        }
        with pytest.raises(RuntimeError, match="Unsupported safetensors dtype"):
            _plan(loader, headers)

    def test_plan_is_offset_ordered_within_a_shard(self):
        loader = _loader()
        headers = _headers(2, tensors_per_shard=4, tensor_size=1024)
        plan = _plan(loader, headers, MX_OBJ_GROUP_MB="64")
        for g in plan:
            offs = [m.rel_offset for m in g.members]
            assert offs == sorted(offs)


# ---------------------------------------------------------------------------
# Staging budget
# ---------------------------------------------------------------------------


class TestStagingBudget:
    """Sized by what keeps the backend fed, not by what VRAM is free."""

    def test_does_not_scale_with_the_largest_descriptor(self):
        # A single oversized tensor must not set the budget for the whole model.
        # Gemma-3-27B's 2.6 GiB embedding against a 159 MiB mean group produced a
        # 21 GiB budget under the old 8x-largest rule.
        from modelexpress.obj_loader import _MIN_STAGING_BYTES

        loader = _loader()
        a, b, c = _cuda_mem(free=64 * 1024 ** 3)
        with a, b, c, patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_STAGING_MB", None)
            small = loader._resolve_staging_budget(159 * 1024 * 1024)
            huge = loader._resolve_staging_budget(2688 * 1024 * 1024)
        assert small == _MIN_STAGING_BYTES
        assert huge == _MIN_STAGING_BYTES, "an outlier tensor inflated the budget"

    def test_raised_to_admit_an_oversized_group(self):
        # Beyond the floor the budget must still hold one whole group, since the
        # feed loop admits one regardless.
        loader = _loader()
        giant = 8 * 1024 ** 3
        a, b, c = _cuda_mem(free=128 * 1024 ** 3)
        with a, b, c, patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_STAGING_MB", None)
            assert loader._resolve_staging_budget(giant) == giant

    def test_does_not_scale_with_free_vram(self):
        # The old formula took a share of free VRAM; the budget must now be the
        # same on a nearly-empty GPU as on a busy one.
        loader = _loader()
        desc = 8 * 1024 * 1024
        budgets = []
        for free in (8 * 1024 ** 3, 64 * 1024 ** 3, 128 * 1024 ** 3):
            a, b, c = _cuda_mem(free=free)
            with a, b, c, patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("MX_OBJ_STAGING_MB", None)
                budgets.append(loader._resolve_staging_budget(desc))
        assert len(set(budgets)) == 1, f"budget tracked free VRAM: {budgets}"

    def test_floor_applies_to_small_descriptors(self):
        from modelexpress.obj_loader import _MIN_STAGING_BYTES

        loader = _loader()
        a, b, c = _cuda_mem(free=64 * 1024 ** 3)
        with a, b, c, patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_STAGING_MB", None)
            assert loader._resolve_staging_budget(1024) == _MIN_STAGING_BYTES

    def test_clamped_by_free_vram(self):
        from modelexpress.obj_loader import _STAGING_VRAM_CEILING

        loader = _loader()
        free = 4 * 1024 ** 3          # floor of 2 GiB exceeds the 50% ceiling
        a, b, c = _cuda_mem(free=free)
        with a, b, c, patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_STAGING_MB", None)
            budget = loader._resolve_staging_budget(1024)
        assert budget == int(free * _STAGING_VRAM_CEILING)

    def test_explicit_override_wins(self):
        loader = _loader()
        a, b, c = _cuda_mem(free=1024 ** 4)
        with a, b, c, patch.dict("os.environ", {"MX_OBJ_STAGING_MB": "512"}):
            assert loader._resolve_staging_budget(1024) == 512 * 1024 * 1024


# ---------------------------------------------------------------------------
# Feed loop
# ---------------------------------------------------------------------------


class TestFeedLoop:
    """Posts ahead up to the staging budget, yields each group as it lands."""

    def _run(self, loader, headers, budget_bytes, group_mb="64", **kw):
        with patch("modelexpress.obj_loader.is_obj_available", return_value=True), \
             patch.object(loader, "_ensure_obj_manager"), \
             patch.object(loader, "_parse_object_headers", return_value=headers), \
             patch.object(
                 loader, "_resolve_shard_basenames", return_value=list(headers)
             ), \
             patch.object(
                 loader, "_resolve_staging_budget", return_value=budget_bytes
             ), \
             patch.dict("os.environ", {"MX_OBJ_GROUP_MB": group_mb}), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.device", return_value=None):
            return list(loader.load_iter("org/model", "", use_tqdm=False, **kw))

    def test_one_transfer_per_group_not_per_tensor(self):
        manager = _FakeManager()
        loader = _loader(manager)
        # 8 tensors of 1 MiB in one shard, 4 MiB groups -> 2 transfers, 8 tensors.
        headers = _headers(1, tensors_per_shard=8, tensor_size=1024 * 1024)

        out = self._run(loader, headers, budget_bytes=1 << 30, group_mb="4")

        assert len(out) == 8, "every tensor must still be delivered"
        submits = [e for e in manager.events if e[0] == "submit"]
        assert len(submits) == 2, "expected one transfer per group"
        # Each transfer is a single contiguous range covering the whole group.
        for _kind, _h, ranges in submits:
            assert len(ranges) == 1 and len(ranges[0][1]) == 1
            assert ranges[0][1][0][1] == 4 * 1024 * 1024

    def test_yields_named_typed_tensors_from_a_shared_buffer(self):
        manager = _FakeManager()
        loader = _loader(manager)
        headers = {
            "s.safetensors": {
                "w1": {"offset": 8, "size": 32, "dtype": "F32", "shape": [4, 2]},
                "w2": {"offset": 40, "size": 16, "dtype": "F32", "shape": [4]},
            }
        }
        out = self._run(loader, headers, budget_bytes=1 << 30)

        assert [n for n, _ in out] == ["w1", "w2"]
        assert [list(t.shape) for _, t in out] == [[4, 2], [4]]
        assert all(t.dtype == torch.float32 for _, t in out)
        # One transfer covered both, so both views share one storage.
        assert len({t.untyped_storage().data_ptr() for _, t in out}) == 1

    def test_posts_ahead_before_awaiting(self):
        manager = _FakeManager()
        loader = _loader(manager)
        # 5 groups of 1 MiB each (1 MiB target), budget far above -> all posted.
        headers = _headers(1, tensors_per_shard=5, tensor_size=1024 * 1024)

        self._run(loader, headers, budget_bytes=1 << 30, group_mb="1")

        kinds = [e[0] for e in manager.events]
        first_wait = kinds.index("wait")
        assert first_wait == 5, (
            f"expected all 5 submits before the first wait, got {kinds}"
        )

    def test_budget_bounds_outstanding_bytes(self):
        manager = _FakeManager()
        loader = _loader(manager)
        size = 1024 * 1024
        headers = _headers(1, tensors_per_shard=8, tensor_size=size)

        peak = 0
        real_submit = manager.submit_objects

        def tracking_submit(jobs, device):
            nonlocal peak
            handle = real_submit(jobs, device)
            peak = max(peak, len(manager.outstanding()))
            return handle

        manager.submit_objects = tracking_submit
        self._run(loader, headers, budget_bytes=3 * size, group_mb="1")

        assert peak <= 3, f"outstanding transfers peaked at {peak}, budget allowed 3"

    def test_oversized_group_is_still_admitted(self):
        manager = _FakeManager()
        loader = _loader(manager)
        headers = {
            "s.safetensors": {
                "big": {"offset": 8, "size": 4096, "dtype": "U8", "shape": [4096]}
            }
        }
        out = self._run(loader, headers, budget_bytes=16)

        assert [n for n, _ in out] == ["big"]

    def test_drains_everything_on_abandonment(self):
        manager = _FakeManager()
        loader = _loader(manager)
        headers = _headers(1, tensors_per_shard=5, tensor_size=1024 * 1024)

        with patch("modelexpress.obj_loader.is_obj_available", return_value=True), \
             patch.object(loader, "_ensure_obj_manager"), \
             patch.object(loader, "_parse_object_headers", return_value=headers), \
             patch.object(
                 loader, "_resolve_shard_basenames", return_value=list(headers)
             ), \
             patch.object(loader, "_resolve_staging_budget", return_value=1 << 30), \
             patch.dict("os.environ", {"MX_OBJ_GROUP_MB": "1"}), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.device", return_value=None):
            it = loader.load_iter("org/model", "", use_tqdm=False)
            next(it)
            it.close()

        assert manager.outstanding() == [], "posted transfers were left unawaited"
