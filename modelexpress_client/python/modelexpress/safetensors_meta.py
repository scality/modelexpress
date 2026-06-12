# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared safetensors metadata helpers.

Source-agnostic safetensors header parsing used by both the GDS
(file-backed) and OBJ (object-store) loaders. The byte source is supplied
as a ``read_fn(offset, length) -> bytes`` callback so the same parser works
over a file descriptor, an object-store ranged GET, or an in-memory blob.
"""

from __future__ import annotations

import json
import struct
from typing import Callable

import torch

# Complete dtype mapping from the safetensors spec:
# https://huggingface.co/docs/safetensors/metadata_parsing#accepted-dtypes
SAFETENSORS_DTYPE_MAP: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# Sanity cap on the safetensors header so a corrupt length field cannot
# trigger an enormous read.
MAX_HEADER_SIZE = 100 * 1024 * 1024


def parse_safetensors_header(
    read_fn: Callable[[int, int], bytes],
) -> dict[str, dict]:
    """Parse a safetensors header using a generic byte-range reader.

    Args:
        read_fn: ``read_fn(offset, length)`` returns exactly ``length`` bytes
            starting at ``offset`` from the safetensors blob.

    Returns:
        ``{tensor_name: {"offset": int, "size": int, "dtype": str, "shape": list}}``
        where ``offset`` is the absolute byte offset of the tensor data within
        the blob.
    """
    raw = read_fn(0, 8)
    if len(raw) < 8:
        raise RuntimeError("Invalid safetensors blob: truncated header size")

    header_size = struct.unpack("<Q", raw)[0]
    if header_size > MAX_HEADER_SIZE:
        raise RuntimeError(f"Safetensors header too large ({header_size} bytes)")

    header = json.loads(read_fn(8, header_size))
    data_start = 8 + header_size

    result: dict[str, dict] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        offsets = info["data_offsets"]
        result[name] = {
            "offset": data_start + offsets[0],
            "size": offsets[1] - offsets[0],
            "dtype": info["dtype"],
            "shape": info["shape"],
        }
    return result
