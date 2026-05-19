# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from pathlib import Path

import torch

from vllm.v1.attention.backends import flash_attn as flash_attn_backend
from vllm.v1.debug import kv_dump as kv_dump_module


class DummyAttentionLayer:
    def __init__(self) -> None:
        self.layer_name = "model.layers.0.attn"
        self._k_scale = torch.tensor(1.0)
        self._v_scale = torch.tensor(1.0)


def test_kv_dump_writer_writes_payload_and_manifest(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv("VLLM_KV_DUMP_DIR", temp_dir)
        monkeypatch.delenv("VLLM_KV_DUMP_MAX_EVENTS", raising=False)
        monkeypatch.delenv("VLLM_KV_DUMP_LAYERS", raising=False)
        monkeypatch.delenv("VLLM_KV_DUMP_RANKS", raising=False)
        kv_dump_module._writer = None

        key = torch.arange(5 * 2 * 3, dtype=torch.float32).view(5, 2, 3)
        value = key + 100
        slot_mapping = torch.tensor([4, 5, 6], dtype=torch.int64)

        kv_dump_module.dump_kv_cache_write(
            layer_name="model.layers.0.attn",
            backend_name="flash_attn",
            key=key,
            value=value,
            slot_mapping=slot_mapping,
            kv_cache_dtype="auto",
        )

        rank_dirs = list(Path(temp_dir).glob("rank_*"))
        assert len(rank_dirs) == 1
        manifest = rank_dirs[0] / "manifest.json"
        assert manifest.exists()

        event_files = list(rank_dirs[0].glob("model.layers.0.attn/event_*.pt"))
        assert len(event_files) == 1
        payload = torch.load(event_files[0], weights_only=False)
        assert payload["key"].shape == (3, 2, 3)
        assert payload["value"].shape == (3, 2, 3)

        kv_dump_module._writer = None


def test_kv_dump_writer_respects_layer_filter(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv("VLLM_KV_DUMP_DIR", temp_dir)
        monkeypatch.setenv("VLLM_KV_DUMP_LAYERS", "model.layers.1.attn")
        kv_dump_module._writer = None

        tensor = torch.randn(4, 2, 3)
        slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
        kv_dump_module.dump_kv_cache_write(
            layer_name="model.layers.0.attn",
            backend_name="flash_attn",
            key=tensor,
            value=tensor,
            slot_mapping=slot_mapping,
            kv_cache_dtype="auto",
        )

        rank_dirs = list(Path(temp_dir).glob("rank_*"))
        if rank_dirs:
            assert not list(rank_dirs[0].glob("**/event_*.pt"))

        kv_dump_module._writer = None


def test_flash_attention_backend_dumps_before_cache_update(monkeypatch):
    captured = {}

    def fake_dump(**kwargs):
        captured.update(kwargs)

    def fake_kernel(*args, **kwargs):
        captured["kernel_called"] = True

    monkeypatch.setattr(
        flash_attn_backend, "dump_kv_cache_write", fake_dump, raising=False
    )
    monkeypatch.setattr(
        flash_attn_backend, "reshape_and_cache_flash", fake_kernel, raising=False
    )

    impl = flash_attn_backend.FlashAttentionImpl(
        num_heads=2,
        head_size=8,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    layer = DummyAttentionLayer()
    key = torch.randn(5, 2, 8)
    value = torch.randn(5, 2, 8)
    kv_cache = torch.zeros(2, 4, 16, 2, 8)
    slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int64)

    impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    assert captured["layer_name"] == layer.layer_name
    assert captured["backend_name"] == "flash_attn"
    assert captured["kv_cache_dtype"] == "auto"
    assert torch.equal(captured["slot_mapping"], slot_mapping)
    assert captured["kernel_called"] is True
