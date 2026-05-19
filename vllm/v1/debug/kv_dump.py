# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug helpers for dumping KV cache writes during runtime."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import torch

_KV_DUMP_DIR_ENV = "VLLM_KV_DUMP_DIR"
_KV_DUMP_FORMAT_ENV = "VLLM_KV_DUMP_FORMAT"
_KV_DUMP_MAX_EVENTS_ENV = "VLLM_KV_DUMP_MAX_EVENTS"
_KV_DUMP_LAYERS_ENV = "VLLM_KV_DUMP_LAYERS"
_KV_DUMP_RANKS_ENV = "VLLM_KV_DUMP_RANKS"
_DEFAULT_DUMP_FORMAT = "pt"

_writer_lock = threading.Lock()
_writer: "KVDumpWriter | None" = None


def _parse_csv_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    for name in ("RANK", "LOCAL_RANK"):
        raw = os.getenv(name)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                continue
    return 0


def _sanitize_layer_name(layer_name: str) -> str:
    return layer_name.replace("/", "_")


class KVDumpWriter:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = _get_rank()
        self.pid = os.getpid()
        self.rank_dir = self.output_dir / f"rank_{self.rank:05d}_pid_{self.pid}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.dump_format = os.getenv(_KV_DUMP_FORMAT_ENV, _DEFAULT_DUMP_FORMAT)
        if self.dump_format != "pt":
            raise ValueError(
                f"Unsupported KV dump format: {self.dump_format}. Only pt is supported."
            )

        raw_max_events = os.getenv(_KV_DUMP_MAX_EVENTS_ENV)
        self.max_events = int(raw_max_events) if raw_max_events else None
        self.layer_filter = _parse_csv_set(os.getenv(_KV_DUMP_LAYERS_ENV))
        raw_ranks = _parse_csv_set(os.getenv(_KV_DUMP_RANKS_ENV))
        self.rank_filter = {int(rank) for rank in raw_ranks} if raw_ranks else None
        self._next_event_id = 0
        self._events_written = 0
        self._lock = threading.Lock()
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "format_version": 1,
            "dump_format": self.dump_format,
            "rank": self.rank,
            "pid": self.pid,
            "created_at_ns": time.time_ns(),
            "layer_filter": sorted(self.layer_filter) if self.layer_filter else None,
            "rank_filter": sorted(self.rank_filter) if self.rank_filter else None,
            "max_events": self.max_events,
        }
        manifest_path = self.rank_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def should_dump(self, layer_name: str) -> bool:
        if self.rank_filter is not None and self.rank not in self.rank_filter:
            return False
        if self.layer_filter is not None and layer_name not in self.layer_filter:
            return False
        if self.max_events is not None and self._events_written >= self.max_events:
            return False
        return True

    def dump_write(
        self,
        *,
        layer_name: str,
        backend_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str | None,
    ) -> None:
        if not self.should_dump(layer_name):
            return

        slot_mapping = slot_mapping.reshape(-1)
        num_tokens = int(slot_mapping.numel())
        if num_tokens == 0:
            return

        key_to_dump = key[:num_tokens].detach().cpu()
        value_to_dump = value[:num_tokens].detach().cpu()

        with self._lock:
            if self.max_events is not None and self._events_written >= self.max_events:
                return
            event_id = self._next_event_id
            self._next_event_id += 1
            self._events_written += 1

        layer_dir = self.rank_dir / _sanitize_layer_name(layer_name)
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / f"event_{event_id:08d}.pt"
        payload = {
            "key": key_to_dump,
            "value": value_to_dump,
        }
        torch.save(payload, path)


def get_kv_dump_writer() -> KVDumpWriter | None:
    global _writer
    if _writer is not None:
        return _writer

    output_dir = os.getenv(_KV_DUMP_DIR_ENV)
    if not output_dir:
        return None

    with _writer_lock:
        if _writer is None:
            _writer = KVDumpWriter(output_dir)
    return _writer


def dump_kv_cache_write(
    *,
    layer_name: str,
    backend_name: str,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str | None,
) -> None:
    writer = get_kv_dump_writer()
    if writer is None:
        return
    writer.dump_write(
        layer_name=layer_name,
        backend_name=backend_name,
        key=key,
        value=value,
        slot_mapping=slot_mapping,
        kv_cache_dtype=kv_cache_dtype,
    )
