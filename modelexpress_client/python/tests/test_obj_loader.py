# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OBJ loader and transfer manager."""

import gc
import json
import struct
import weakref
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

    def test_unset_returns_only_the_dram_declaration(self):
        # Our host buffers hold metadata, which no backend default can infer, so
        # this one key is declared even with nothing configured.
        from modelexpress.obj_transfer import obj_backend_params
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_PARAMS", None)
            assert obj_backend_params() == {"dram_rdma": "false"}

    def test_parses_and_stringifies(self):
        from modelexpress.obj_transfer import obj_backend_params
        raw = '{"accelerated": true, "type": "vendor_engine", "num_threads": 8}'
        with patch.dict("os.environ", {"MX_OBJ_PARAMS": raw}):
            params = obj_backend_params()
        assert params == {
            "accelerated": "True",
            "type": "vendor_engine",
            "num_threads": "8",
            "dram_rdma": "false",
        }

    def test_explicit_dram_rdma_is_not_overridden(self):
        # The operator's value wins; this is how the RDMA and HTTP header paths
        # get compared against each other.
        from modelexpress.obj_transfer import obj_backend_params
        with patch.dict("os.environ", {"MX_OBJ_PARAMS": '{"dram_rdma": "true"}'}):
            assert obj_backend_params()["dram_rdma"] == "true"

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
    """Stand-in for ObjTransferManager recording pool and submit/wait ordering.

    Records an `events` list of ("submit", handle, key, offset, size, pool_offset)
    and ("wait", handle) so a test can assert not just what was requested but when
    it was awaited relative to other submissions -- and, for the pooled path, which
    region of the pool each transfer landed in.
    """

    def __init__(self):
        self.events = []
        self._seq = 0
        self._ranges = {}
        self.pool = None
        self.pool_opens = 0
        self.registered_objects = None
        self.closed = False

    # -- pooled path ---------------------------------------------------------

    def open_pool(self, pool_bytes, device):
        self.pool_opens += 1
        self.pool = torch.zeros(pool_bytes, dtype=torch.uint8)
        self.events.append(("open_pool", pool_bytes))

    def register_objects(self, object_sizes):
        self.registered_objects = dict(object_sizes)
        self.events.append(("register_objects", len(object_sizes)))

    def submit_pooled(self, object_key, obj_offset, size, pool_offset):
        self._seq += 1
        handle = _FakeHandle(f"h{self._seq}")
        self.events.append(
            ("submit", handle.name, object_key, obj_offset, size, pool_offset)
        )
        self._ranges[handle.name] = (object_key, obj_offset, size, pool_offset)
        return handle

    def close_pool(self):
        self.closed = True
        self.events.append(("close_pool",))

    # -- shared --------------------------------------------------------------

    def wait_objects(self, handle):
        self.events.append(("wait", handle.name))
        return []

    def outstanding(self):
        """Handles submitted but not yet awaited."""
        submitted = [e[1] for e in self.events if e[0] == "submit"]
        awaited = {e[1] for e in self.events if e[0] == "wait"}
        return [h for h in submitted if h not in awaited]

    def submitted(self):
        """[(key, obj_offset, size, pool_offset), ...] in submission order."""
        return [
            (e[2], e[3], e[4], e[5]) for e in self.events if e[0] == "submit"
        ]

    def live_regions(self):
        """Pool regions of transfers submitted but not yet awaited."""
        awaited = {e[1] for e in self.events if e[0] == "wait"}
        return [
            (e[5], e[5] + e[4])
            for e in self.events
            if e[0] == "submit" and e[1] not in awaited
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
        return loader._build_plan("", list(headers))


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
        # The VRAM clamp must not cut the budget below one group: the feed loop
        # admits one regardless, so a budget that cannot hold it is a lie. The
        # floor now sits at the registration limit, so the clamp path is the only
        # one that can leave the budget below a group.
        loader = _loader()
        big = 3 * 1024 ** 3               # over the 50% ceiling of 4 GiB free
        a, b, c = _cuda_mem(free=4 * 1024 ** 3)
        with a, b, c, patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_STAGING_MB", None)
            assert loader._resolve_staging_budget(big) == big

    def test_never_exceeds_the_registration_limit(self):
        # The pool is a single registration, so no input -- an oversized group,
        # an override, or abundant VRAM -- may push the budget past what the
        # backend will register. Asking for more fails the whole load.
        from modelexpress.obj_transfer import MAX_REG_BYTES

        loader = _loader()
        a, b, c = _cuda_mem(free=512 * 1024 ** 3)
        with a, b, c, patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_STAGING_MB", None)
            assert loader._resolve_staging_budget(64 * 1024 ** 3) == MAX_REG_BYTES
        a, b, c = _cuda_mem(free=512 * 1024 ** 3)
        with a, b, c, patch.dict("os.environ", {"MX_OBJ_STAGING_MB": "65536"}):
            assert loader._resolve_staging_budget(1024) == MAX_REG_BYTES

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
# Pool ring allocator
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Stand-in for torch.cuda.Event with query() under test control."""

    def __init__(self):
        self.done = False
        self.synchronized = False

    def record(self):
        pass

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronized = True
        self.done = True


class TestPoolRing:
    """Regions are reused only once their consumer has demonstrably finished."""

    def _ring(self, size):
        from modelexpress.obj_loader import _PoolRing
        return _PoolRing(size)

    def test_allocates_sequentially(self):
        ring = self._ring(1000)
        assert ring.alloc(300) == 0
        assert ring.alloc(300) == 300
        assert ring.alloc(300) == 600

    def test_rejects_a_group_larger_than_the_pool(self):
        ring = self._ring(100)
        with pytest.raises(RuntimeError, match="exceeds"):
            ring.alloc(101)

    def test_blocks_on_in_flight_regions(self):
        # Nothing consumed yet, so the ring cannot free anything itself.
        ring = self._ring(1000)
        ring.alloc(600)
        ring.alloc(400)
        assert ring.alloc(100) is None, "handed out memory still being written"

    def test_reclaims_a_completed_region_without_waiting(self):
        # The fast path: the consumer has finished, so the region is reclaimed by
        # polling the event and the allocation wraps into it.
        ring = self._ring(1000)
        first = ring.alloc(600)
        ring.alloc(400)
        with patch("torch.cuda.Event", _FakeEvent):
            ring.consumed(first)
        event = ring._regions[0][2]
        event.done = True

        assert ring.alloc(100) == 0, "should wrap into the reclaimed region"
        assert not event.synchronized, "should poll, not block, when already done"

    def test_waits_rather_than_failing_when_only_consumed_regions_block(self):
        # A consumed region whose event has not fired must be waited on, not
        # reported as a failure: the caller has nothing left to drain.
        ring = self._ring(1000)
        first = ring.alloc(1000)
        with patch("torch.cuda.Event", _FakeEvent):
            ring.consumed(first)
        event = ring._regions[0][2]
        assert ring.alloc(500) == 0
        assert event.synchronized, "should have waited on the consumer's copies"

    def test_regions_never_overlap(self):
        # Drive a long sequence of varied sizes and assert the invariant that
        # matters: no live region ever overlaps another.
        import random

        rnd = random.Random(1234)
        ring = self._ring(4096)
        live = []
        with patch("torch.cuda.Event", _FakeEvent):
            for _ in range(400):
                size = rnd.randint(1, 900)
                off = ring.alloc(size)
                if off is None:
                    if not live:
                        continue
                    done_off = live.pop(0)[0]
                    ring.consumed(done_off)
                    for r in ring._regions:
                        if r[0] == done_off and r[2] is not None:
                            r[2].done = True
                    continue
                for lo, hi in live:
                    assert off >= hi or off + size <= lo, (
                        f"region [{off},{off + size}) overlaps live [{lo},{hi})"
                    )
                live.append((off, off + size))
                if len(live) > 3:
                    done_off = live.pop(0)[0]
                    ring.consumed(done_off)
                    for r in ring._regions:
                        if r[0] == done_off and r[2] is not None:
                            r[2].done = True


# ---------------------------------------------------------------------------
# Feed loop
# ---------------------------------------------------------------------------


class TestFeedLoop:
    """One pooled transfer per group; the pool is registered once."""

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
             patch("torch.cuda.Event", _FakeEvent), \
             patch("torch.device", return_value=None):
            return list(loader.load_iter("org/model", "", use_tqdm=False, **kw))

    def test_pool_opened_once_and_keys_registered_once(self):
        manager = _FakeManager()
        loader = _loader(manager)
        headers = _headers(3, tensors_per_shard=4, tensor_size=1024 * 1024)

        self._run(loader, headers, budget_bytes=1 << 30, group_mb="4")

        assert manager.pool_opens == 1, "the pool must be registered once per load"
        assert len(manager.registered_objects) == 3, "one devId per shard object"
        assert manager.closed, "the pool must be released"
        # No per-transfer registration: submit_objects is never used on this path.
        assert not any(e[0] == "register" for e in manager.events)

    def test_one_transfer_per_group_into_distinct_pool_regions(self):
        manager = _FakeManager()
        loader = _loader(manager)
        headers = _headers(1, tensors_per_shard=8, tensor_size=1024 * 1024)

        out = self._run(loader, headers, budget_bytes=1 << 30, group_mb="4")

        assert len(out) == 8
        submitted = manager.submitted()
        assert len(submitted) == 2, "expected one transfer per group"
        # Regions must not overlap while both are outstanding.
        (_, _, s0, p0), (_, _, s1, p1) = submitted
        assert p1 >= p0 + s0 or p0 >= p1 + s1

    def test_yields_named_typed_tensors_from_the_pool(self):
        manager = _FakeManager()
        loader = _loader(manager)
        headers = {
            "s.safetensors": {
                "w1": {"offset": 8, "size": 32, "dtype": "F32", "shape": [4, 2]},
                "w2": {"offset": 40, "size": 16, "dtype": "F32", "shape": [4]},
            }
        }
        out = self._run(loader, headers, budget_bytes=1 << 20)

        assert [n for n, _ in out] == ["w1", "w2"]
        assert [list(t.shape) for _, t in out] == [[4, 2], [4]]
        assert all(t.dtype == torch.float32 for _, t in out)
        # Both are views of the pool, not copies.
        pool_ptr = manager.pool.data_ptr()
        pool_end = pool_ptr + manager.pool.numel()
        for _, t in out:
            assert pool_ptr <= t.data_ptr() < pool_end

    def test_posts_ahead_before_awaiting(self):
        manager = _FakeManager()
        loader = _loader(manager)
        headers = _headers(1, tensors_per_shard=5, tensor_size=1024 * 1024)

        self._run(loader, headers, budget_bytes=1 << 30, group_mb="1")

        kinds = [e[0] for e in manager.events if e[0] in ("submit", "wait")]
        assert kinds.index("wait") == 5, (
            f"expected all 5 submits before the first wait, got {kinds}"
        )

    def test_pool_bounds_outstanding_bytes(self):
        manager = _FakeManager()
        loader = _loader(manager)
        size = 1024 * 1024
        headers = _headers(1, tensors_per_shard=8, tensor_size=size)

        self._run(loader, headers, budget_bytes=3 * size, group_mb="1")

        # Peak live pool bytes must never exceed the pool.
        peak = 0
        live = {}
        for e in manager.events:
            if e[0] == "submit":
                live[e[1]] = e[4]
            elif e[0] == "wait":
                live.pop(e[1], None)
            peak = max(peak, sum(live.values()))
        assert peak <= 3 * size, f"outstanding peaked at {peak}, pool was {3 * size}"

    def test_oversized_group_is_still_admitted(self):
        manager = _FakeManager()
        loader = _loader(manager)
        headers = {
            "s.safetensors": {
                "big": {"offset": 8, "size": 4096, "dtype": "U8", "shape": [4096]}
            }
        }
        # Budget exactly the group: the ring must admit it rather than deadlock.
        out = self._run(loader, headers, budget_bytes=4096)

        assert [n for n, _ in out] == ["big"]

    def test_drains_and_closes_on_abandonment(self):
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
             patch("torch.cuda.Event", _FakeEvent), \
             patch("torch.device", return_value=None):
            it = loader.load_iter("org/model", "", use_tqdm=False)
            next(it)
            it.close()

        assert manager.outstanding() == [], "posted transfers were left unawaited"
        assert manager.closed, "the pool must be released even on abandonment"


# ---------------------------------------------------------------------------
# Pool registration
# ---------------------------------------------------------------------------


class _RegAgent:
    """Minimal agent stub: register_memory either succeeds or raises."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def register_memory(self, regions, mem_type):
        self.calls += 1
        if self.fail:
            raise RuntimeError("NIXL_ERR_BACKEND")
        return object()


class TestOpenPool:
    """The pool is one registration, so its size and failure path both matter."""

    @staticmethod
    def _manager(agent):
        from modelexpress.obj_transfer import ObjTransferManager
        manager = ObjTransferManager(agent_name="mx-obj-test", params={})
        manager._agent = agent
        manager._device_id = 0
        return manager

    def test_rejects_a_pool_the_backend_cannot_register(self):
        from modelexpress.obj_transfer import MAX_REG_BYTES

        agent = _RegAgent()
        manager = self._manager(agent)
        with pytest.raises(RuntimeError, match="registration limit"):
            manager.open_pool(MAX_REG_BYTES + 1, torch.device("cpu"))
        assert agent.calls == 0, "an unregistrable size must not reach the backend"
        assert manager.pool is None

    def test_releases_the_buffer_when_registration_fails(self):
        # The buffer is a large share of VRAM and the caller's next move is a
        # fallback loader that needs it, so nothing may stay referenced.
        agent = _RegAgent(fail=True)
        manager = self._manager(agent)
        buffers = []

        real_empty = torch.empty

        def tracking_empty(*args, **kwargs):
            kwargs.pop("device", None)
            tensor = real_empty(*args, **kwargs)
            buffers.append(weakref.ref(tensor))
            return tensor

        with patch("torch.empty", tracking_empty), \
             patch("torch.cuda.empty_cache"):
            with pytest.raises(RuntimeError, match="NIXL_ERR_BACKEND"):
                manager.open_pool(1024, torch.device("cpu"))

        assert manager.pool is None
        assert len(buffers) == 1
        gc.collect()
        assert buffers[0]() is None, "the staging buffer outlived the failure"


# ---------------------------------------------------------------------------
# Speculative header probe
# ---------------------------------------------------------------------------


def _safetensors_blob(tensors_per_shard, tensor_size=1024, dtype="U8", pad=0):
    """A real safetensors prefix: u64 length, JSON header, then `pad` data bytes."""
    header = {
        f"t{i}": {
            "dtype": dtype,
            "shape": [tensor_size],
            "data_offsets": [i * tensor_size, (i + 1) * tensor_size],
        }
        for i in range(tensors_per_shard)
    }
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\x00" * pad


class _ReadRecorder:
    """Manager stand-in serving byte ranges out of in-memory objects."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.batches: list[list[tuple[str, int, int]]] = []

    def read_ranges_to_host(self, reads):
        self.batches.append(list(reads))
        out = []
        for key, offset, size in reads:
            blob = self.objects[key]
            # Matches the real contract: always `size` bytes back, with the tail
            # left zero when the range reaches past the end of the object (which
            # the connector answers as a complete but shorter 206).
            got = blob[offset:offset + size]
            out.append(got + b"\x00" * (size - len(got)))
        return out


class TestHeaderProbe:
    """One round trip when the header fits the probe; a second only when it does not."""

    def test_single_batch_when_headers_fit_the_probe(self):
        from modelexpress.obj_loader import _HEADER_PROBE_BYTES

        blob = _safetensors_blob(4, pad=4096)
        manager = _ReadRecorder({"a.safetensors": blob, "b.safetensors": blob})
        loader = _loader(manager)

        headers = loader._parse_object_headers(["a.safetensors", "b.safetensors"])

        assert len(manager.batches) == 1, "a fitting header must not need a second read"
        assert manager.batches[0] == [
            ("a.safetensors", 0, _HEADER_PROBE_BYTES),
            ("b.safetensors", 0, _HEADER_PROBE_BYTES),
        ]
        assert sorted(headers) == ["a.safetensors", "b.safetensors"]
        assert sorted(headers["a.safetensors"]) == ["t0", "t1", "t2", "t3"]
        assert headers["a.safetensors"]["t0"]["size"] == 1024

    def test_short_probe_response_still_parses(self):
        # The object is far smaller than the probe, so the read comes back short.
        # Nothing needed is lost: a valid object holds its whole header.
        blob = _safetensors_blob(2, pad=16)
        manager = _ReadRecorder({"tiny.safetensors": blob})
        loader = _loader(manager)

        headers = loader._parse_object_headers(["tiny.safetensors"])

        assert len(manager.batches) == 1
        assert sorted(headers["tiny.safetensors"]) == ["t0", "t1"]

    def test_second_batch_only_for_shards_that_overflow(self):
        from modelexpress.obj_loader import _HEADER_PROBE_BYTES

        small = _safetensors_blob(2, pad=64)
        # Enough tensors that the JSON header cannot fit in the probe.
        big = _safetensors_blob(2000, pad=64)
        assert len(big) > _HEADER_PROBE_BYTES, "fixture must actually overflow"
        manager = _ReadRecorder({"small.safetensors": small, "big.safetensors": big})
        loader = _loader(manager)

        headers = loader._parse_object_headers(
            ["small.safetensors", "big.safetensors"]
        )

        assert len(manager.batches) == 2
        assert [k for k, _o, _s in manager.batches[1]] == ["big.safetensors"], \
            "the follow-up must carry only the shards that overflowed"
        assert manager.batches[1][0][1] == 8, "follow-up reads the JSON, not the u64"
        assert len(headers["small.safetensors"]) == 2
        assert len(headers["big.safetensors"]) == 2000

    def test_empty_object_raises_rather_than_parsing_zeros(self):
        # A zero-filled buffer decodes as header_size 0, which must not be mistaken
        # for a valid empty header, and must say so rather than failing in json.
        manager = _ReadRecorder({"empty.safetensors": b""})
        loader = _loader(manager)
        with pytest.raises(RuntimeError, match="no safetensors header"):
            loader._parse_object_headers(["empty.safetensors"])

    def test_no_keys_reads_nothing(self):
        manager = _ReadRecorder({})
        loader = _loader(manager)
        assert loader._parse_object_headers([]) == {}
        assert manager.batches == []


# ---------------------------------------------------------------------------
# read_ranges_to_host
# ---------------------------------------------------------------------------


class _HostReadAgent:
    """Agent stub that fulfils a DRAM transfer by writing into the real buffer.

    The DRAM descriptors carry genuine pointers into the manager's tensor, so this
    exercises the actual packing arithmetic rather than a model of it.
    """

    name = "stub"

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.registrations: list[tuple[str, int]] = []  # (mem_type, region count)
        self.dram_descs: list[tuple[int, int, int]] = []
        self.obj_descs: list[tuple[int, int, int]] = []
        self.dev_to_key: dict[int, str] = {}
        self.released = 0

    def register_memory(self, regions, mem_type):
        self.registrations.append((mem_type, len(regions)))
        if mem_type == "OBJ":
            for _addr, _len, dev_id, key in regions:
                self.dev_to_key[dev_id] = key
        return f"{mem_type}-reg-{len(self.registrations)}"

    def deregister_memory(self, reg):
        self.released += 1

    def get_xfer_descs(self, descs, mem_type):
        if mem_type == "DRAM":
            self.dram_descs = list(descs)
        else:
            self.obj_descs = list(descs)
        return (mem_type, list(descs))

    def initialize_xfer(self, op, local, remote, agent_name):
        return ("xfer", local, remote)

    def transfer(self, xfer):
        import ctypes
        for (addr, size, _dev), (obj_off, obj_size, dev_id) in zip(
            self.dram_descs, self.obj_descs, strict=True
        ):
            assert size == obj_size
            blob = self.objects[self.dev_to_key[dev_id]]
            payload = blob[obj_off:obj_off + size]
            if payload:
                ctypes.memmove(addr, payload, len(payload))
        return "DONE"

    def check_xfer_state(self, xfer):
        return "DONE"

    def release_xfer_handle(self, xfer):
        self.released += 1


class TestReadRangesToHost:
    """One buffer, one DRAM registration, bytes back in the order asked for."""

    @staticmethod
    def _manager(agent):
        from modelexpress.obj_transfer import ObjTransferManager
        mgr = ObjTransferManager(agent_name="test", params={})
        mgr._agent = agent
        mgr._device_id = 0
        return mgr

    def test_returns_each_range_in_order(self):
        objects = {
            "a": b"".join(bytes([i]) * 10 for i in range(1, 4)),
            "b": bytes(range(64, 128)),
        }
        agent = _HostReadAgent(objects)
        mgr = self._manager(agent)

        out = mgr.read_ranges_to_host(
            [("a", 0, 10), ("b", 4, 8), ("a", 20, 10)]
        )

        assert out == [
            objects["a"][0:10],
            objects["b"][4:12],
            objects["a"][20:30],
        ]

    def test_one_dram_registration_regardless_of_range_count(self):
        objects = {f"k{i}": bytes([i]) * 32 for i in range(20)}
        agent = _HostReadAgent(objects)
        mgr = self._manager(agent)

        mgr.read_ranges_to_host([(f"k{i}", 0, 32) for i in range(20)])

        dram = [r for r in agent.registrations if r[0] == "DRAM"]
        obj = [r for r in agent.registrations if r[0] == "OBJ"]
        assert dram == [("DRAM", 1)], "the whole buffer must be one registration"
        assert obj == [("OBJ", 20)], "OBJ keys are one call, one region per key"
        assert len(agent.dram_descs) == 20, "one descriptor pair per range"

    def test_ranges_are_aligned_and_non_overlapping_in_the_buffer(self):
        from modelexpress.obj_transfer import _HOST_READ_ALIGN

        objects = {"a": bytes(200), "b": bytes(200), "c": bytes(200)}
        agent = _HostReadAgent(objects)
        mgr = self._manager(agent)

        # Sizes chosen so naive packing would leave them unaligned.
        mgr.read_ranges_to_host([("a", 0, 7), ("b", 0, 3), ("c", 0, 100)])

        base = min(addr for addr, _s, _d in agent.dram_descs)
        spans = [(addr - base, size) for addr, size, _d in agent.dram_descs]
        for offset, _size in spans:
            assert offset % _HOST_READ_ALIGN == 0, f"unaligned pack offset {offset}"
        spans.sort()
        for (off_a, size_a), (off_b, _size_b) in zip(spans, spans[1:], strict=False):
            assert off_a + size_a <= off_b, "packed ranges overlap"

    def test_distinct_dev_ids_so_keys_do_not_collide(self):
        # The backend resolves an object key from the remote descriptor's devId, so
        # two ranges sharing one devId would collapse onto a single key.
        objects = {"a": bytes(64), "b": bytes(64)}
        agent = _HostReadAgent(objects)
        mgr = self._manager(agent)

        mgr.read_ranges_to_host([("a", 0, 8), ("b", 0, 8), ("a", 8, 8)])

        dev_ids = [dev for _off, _size, dev in agent.obj_descs]
        assert len(set(dev_ids)) == 3, f"devIds must be distinct, got {dev_ids}"

    def test_registrations_released_even_on_failure(self):
        objects = {"a": bytes(64)}
        agent = _HostReadAgent(objects)
        agent.transfer = lambda xfer: "ERR"
        mgr = self._manager(agent)

        with pytest.raises(RuntimeError, match="host read failed"):
            mgr.read_ranges_to_host([("a", 0, 8)])

        assert agent.released >= 2, "OBJ and DRAM registrations must both be freed"

    def test_no_reads_is_a_no_op(self):
        agent = _HostReadAgent({})
        mgr = self._manager(agent)
        assert mgr.read_ranges_to_host([]) == []
        assert agent.registrations == []


class TestReadRangesToHostFailurePaths:
    """Neither registration may outlive a failure of the other."""

    def test_obj_registration_freed_when_the_buffer_registration_fails(self):
        from modelexpress.obj_transfer import ObjTransferManager

        freed = []

        class _Agent:
            name = "stub"

            def register_memory(self, regions, mem_type):
                if mem_type == "DRAM":
                    raise RuntimeError("NIXL_ERR_BACKEND")
                return "obj-reg"

            def deregister_memory(self, reg):
                freed.append(reg)

        mgr = ObjTransferManager(agent_name="test", params={})
        mgr._agent = _Agent()
        mgr._device_id = 0

        with pytest.raises(RuntimeError, match="NIXL_ERR_BACKEND"):
            mgr.read_ranges_to_host([("a", 0, 8)])

        assert freed == ["obj-reg"], "the OBJ keys were left registered"


# ---------------------------------------------------------------------------
# Issue order across objects
# ---------------------------------------------------------------------------


class TestInterleaveByObject:
    """Round-robin across objects, preserving each object's own offset order."""

    @staticmethod
    def _groups(spec):
        """spec: {object_key: [offset, ...]} -> plan groups in object-major order."""
        from modelexpress.obj_loader import _PlannedGroup
        out = []
        for key, offsets in spec.items():
            for off in offsets:
                out.append(
                    _PlannedGroup(object_key=key, offset=off, size=1024, members=[])
                )
        return out

    def test_round_robin_across_objects(self):
        from modelexpress.obj_loader import _interleave_by_object

        groups = self._groups({"a": [0, 10, 20], "b": [0, 10, 20], "c": [0, 10, 20]})
        out = _interleave_by_object(groups)

        assert [g.object_key for g in out] == [
            "a", "b", "c", "a", "b", "c", "a", "b", "c"
        ]

    def test_offsets_stay_ascending_within_each_object(self):
        # A server reading ahead inside one object must still see a forward scan.
        from modelexpress.obj_loader import _interleave_by_object

        groups = self._groups({"a": [0, 10, 20, 30], "b": [5, 15]})
        out = _interleave_by_object(groups)

        for key in ("a", "b"):
            offsets = [g.offset for g in out if g.object_key == key]
            assert offsets == sorted(offsets), f"{key} lost its forward order"

    def test_uneven_counts_drain_the_longer_object_last(self):
        from modelexpress.obj_loader import _interleave_by_object

        groups = self._groups({"a": [0, 1, 2, 3], "b": [0]})
        out = _interleave_by_object(groups)

        assert [g.object_key for g in out] == ["a", "b", "a", "a", "a"]
        assert len(out) == len(groups), "no group may be dropped or duplicated"

    def test_single_object_is_unchanged(self):
        from modelexpress.obj_loader import _interleave_by_object

        groups = self._groups({"only": [0, 10, 20]})
        assert _interleave_by_object(groups) == groups

    def test_empty_and_single_group_are_safe(self):
        from modelexpress.obj_loader import _interleave_by_object

        assert _interleave_by_object([]) == []
        one = self._groups({"a": [0]})
        assert _interleave_by_object(one) == one

    def test_every_group_is_preserved_exactly_once(self):
        from modelexpress.obj_loader import _interleave_by_object

        spec = {f"obj{i}": list(range(0, 10 * (i + 1), 10)) for i in range(15)}
        groups = self._groups(spec)
        out = _interleave_by_object(groups)

        assert sorted(out, key=lambda g: (g.object_key, g.offset)) == sorted(
            groups, key=lambda g: (g.object_key, g.offset)
        )

    def test_disabled_by_default_and_enabled_by_env(self):
        from modelexpress.obj_loader import _interleave_enabled

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_OBJ_INTERLEAVE", None)
            assert _interleave_enabled() is False
        with patch.dict("os.environ", {"MX_OBJ_INTERLEAVE": "1"}):
            assert _interleave_enabled() is True
        with patch.dict("os.environ", {"MX_OBJ_INTERLEAVE": "0"}):
            assert _interleave_enabled() is False

    def test_plan_order_follows_the_switch(self):
        loader = _loader()
        headers = _headers(3, tensors_per_shard=2, tensor_size=1024 * 1024)

        object_major = _plan(loader, headers, MX_OBJ_GROUP_MB="1")
        with patch.dict("os.environ", {"MX_OBJ_INTERLEAVE": "1"}):
            interleaved = _plan(loader, headers, MX_OBJ_GROUP_MB="1")

        assert len(interleaved) == len(object_major)
        # Object-major repeats a key before moving on; interleaved does not.
        major_keys = [g.object_key for g in object_major]
        inter_keys = [g.object_key for g in interleaved]
        assert major_keys[0] == major_keys[1], f"expected object-major, got {major_keys}"
        assert inter_keys[0] != inter_keys[1], f"expected interleaved, got {inter_keys}"
