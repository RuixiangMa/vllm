# Adapted from https://github.com/microsoft/VibeVoice

import os
import sys
import uuid
from typing import Any, ClassVar, Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import ModelConfig, VllmConfig
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    ProcessorInputs
)

from collections.abc import Mapping, Sequence

from vllm.transformers_utils.processors.vibevoice_asr import (
    AudioNormalizer,
    AUDIO_SAMPLE_RATE,
    COMMON_AUDIO_EXTS,
    load_audio_use_ffmpeg,
    load_audio_bytes_use_ffmpeg,
    VibeVoiceTokenizerEncoderOutput,
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceSemanticTokenizerModel,
)
from vllm.transformers_utils.configs.vibevoice_asr import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceSemanticTokenizerConfig,
)

logger = init_logger(__name__)


class VibeVoiceASRSpeechConnector(nn.Module):
    """Projects speech features to language model hidden dimension for VibeVoice ASR.

    Architecture: fc1 -> RMSNorm -> fc2 (no activation function)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = VibeVoiceASRRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class VibeVoiceASRRMSNorm(nn.Module):
    """RMSNorm layer used in VibeVoiceASRSpeechConnector for VibeVoice audio processing."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VibeVoiceAudioEncoder(nn.Module):
    """
    VibeVoice Audio Encoder module.

    Encapsulates Acoustic/Semantic VAE Tokenizers and projection Connectors.
    Converts raw audio waveforms into embeddings compatible with the language model.

    Features:
        - Streaming support for long audio (>60s by default)
        - Configurable dtype for numerical precision
        - Supports both sampling and deterministic (mean) modes
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        def get_cfg(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        self.acoustic_vae_dim = get_cfg(config, "acoustic_vae_dim", 64)
        self.semantic_vae_dim = get_cfg(config, "semantic_vae_dim", 128)

        decoder_config = get_cfg(config, "decoder_config")
        text_config = get_cfg(config, "text_config")

        target_hidden_size = None

        if decoder_config is not None:
            target_hidden_size = get_cfg(decoder_config, "hidden_size")

        if target_hidden_size is None and text_config is not None:
            target_hidden_size = get_cfg(text_config, "hidden_size")

        if target_hidden_size is None:
            target_hidden_size = get_cfg(config, "hidden_size")

        if target_hidden_size is None:
            print("[VibeVoice] WARN: Could not find hidden_size in config! Defaulting to 3584 (7B).", file=sys.stderr)
            self.hidden_size = 3584
        else:
            self.hidden_size = target_hidden_size

        ac_cfg = get_cfg(config, "acoustic_tokenizer_config")
        sc_cfg = get_cfg(config, "semantic_tokenizer_config")

        if ac_cfg is None or sc_cfg is None:
            raise ValueError("Missing acoustic/semantic tokenizer config in model config")

        # Handle both dict and already-constructed config objects
        if isinstance(ac_cfg, VibeVoiceAcousticTokenizerConfig):
            acoustic_config = ac_cfg
        elif isinstance(ac_cfg, dict):
            acoustic_config = VibeVoiceAcousticTokenizerConfig(**ac_cfg)
        else:
            raise TypeError(f"acoustic_tokenizer_config has unexpected type: {type(ac_cfg)}")

        if isinstance(sc_cfg, VibeVoiceSemanticTokenizerConfig):
            semantic_config = sc_cfg
        elif isinstance(sc_cfg, dict):
            semantic_config = VibeVoiceSemanticTokenizerConfig(**sc_cfg)
        else:
            raise TypeError(f"semantic_tokenizer_config has unexpected type: {type(sc_cfg)}")

        # Tokenizers use float32 for numerical precision
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(acoustic_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(semantic_config)

        # Get audio encoder dtype from config (defaults to float32 for precision)
        root_torch_dtype = get_cfg(config, "torch_dtype", None)
        if root_torch_dtype is not None:
            if isinstance(root_torch_dtype, str):
                self._audio_encoder_dtype = getattr(torch, root_torch_dtype)
            else:
                self._audio_encoder_dtype = root_torch_dtype
        else:
            self._audio_encoder_dtype = torch.float32

        self.acoustic_connector = VibeVoiceASRSpeechConnector(self.acoustic_vae_dim, self.hidden_size)
        self.semantic_connector = VibeVoiceASRSpeechConnector(self.semantic_vae_dim, self.hidden_size)

        self.compress_ratio = get_cfg(config, "speech_tok_compress_ratio", 3200)

        # Streaming controls
        self.sample_rate = get_cfg(config, "target_sample_rate", 24000)

        # Default to True (per requirement): segment + cache inside one forward call.
        self.enable_streaming = get_cfg(config, "enable_streaming", True)
        self.streaming_segment_duration = get_cfg(config, "streaming_segment_duration", 60.0)

        use_mean_env = os.getenv("VIBEVOICE_USE_MEAN", "").strip().lower()
        self.use_sample = use_mean_env not in ("1", "true", "yes")

        self._lm_dtype: torch.dtype = torch.bfloat16

    def _ensure_audio_encoder_dtype(self):
        target_dtype = self._audio_encoder_dtype

        try:
            acoustic_dtype = next(self.acoustic_tokenizer.parameters()).dtype
            if acoustic_dtype != target_dtype:
                self.acoustic_tokenizer = self.acoustic_tokenizer.to(dtype=target_dtype)
                print(
                    f"[VibeVoice] Converted acoustic_tokenizer to {target_dtype} (was {acoustic_dtype})",
                    file=sys.stderr,
                )
        except (StopIteration, AttributeError, TypeError) as exc:
            logger.warning(
                "Failed to ensure acoustic tokenizer dtype during initialization: %s",
                exc,
            )

        try:
            semantic_dtype = next(self.semantic_tokenizer.parameters()).dtype
            if semantic_dtype != target_dtype:
                self.semantic_tokenizer = self.semantic_tokenizer.to(dtype=target_dtype)
                print(
                    f"[VibeVoice] Converted semantic_tokenizer to {target_dtype} (was {semantic_dtype})",
                    file=sys.stderr,
                )
        except (StopIteration, AttributeError, TypeError) as exc:
            logger.warning(
                "Failed to ensure semantic tokenizer dtype during initialization: %s",
                exc,
            )

        try:
            ac_conn_dtype = next(self.acoustic_connector.parameters()).dtype
            if ac_conn_dtype != target_dtype:
                self.acoustic_connector = self.acoustic_connector.to(dtype=target_dtype)
                print(
                    f"[VibeVoice] Converted acoustic_connector to {target_dtype} (was {ac_conn_dtype})", file=sys.stderr
                )
        except (StopIteration, AttributeError, TypeError) as exc:
            logger.warning(
                "Failed to ensure acoustic connector dtype during initialization: %s",
                exc,
            )

        try:
            sc_conn_dtype = next(self.semantic_connector.parameters()).dtype
            if sc_conn_dtype != target_dtype:
                self.semantic_connector = self.semantic_connector.to(dtype=target_dtype)
                print(
                    f"[VibeVoice] Converted semantic_connector to {target_dtype} (was {sc_conn_dtype})", file=sys.stderr
                )
        except (StopIteration, AttributeError, TypeError) as exc:
            logger.warning(
                "Failed to ensure semantic connector dtype during initialization: %s",
                exc,
            )

    def forward(
        self,
        audio: torch.Tensor,
        *,
        use_streaming: bool = True,
        segment_duration_s: float | None = None,
        use_sample: bool | None = None,
    ) -> torch.Tensor:
        """Encode audio with optional streaming for long clips.

        Args:
            audio: Input audio tensor [B, T] or [T]
            use_streaming: Whether to enable segmented encoding for long audio
            segment_duration_s: Segment length in seconds (defaults to 60s)
            use_sample: If True, use sampling for acoustic tokens; if False, use mean
                       Defaults to self.use_sample (controlled by VIBEVOICE_USE_MEAN env var)

        Returns:
            Audio embeddings tensor compatible with the language model
        """
        # Ensure audio encoder components use correct dtype
        self._ensure_audio_encoder_dtype()

        # Audio input should match the audio encoder dtype
        audio = audio.to(dtype=self._audio_encoder_dtype)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Resolve streaming options
        segment_duration = segment_duration_s or self.streaming_segment_duration
        sample_rate = self.sample_rate
        total_samples = audio.shape[-1]
        segment_samples = int(segment_duration * sample_rate)

        use_streaming = use_streaming and self.enable_streaming and total_samples > segment_samples

        # Resolve use_sample flag
        if use_sample is None:
            use_sample = self.use_sample

        with torch.no_grad():
            if not use_streaming:
                acoustic_input = audio.unsqueeze(1)
                acoustic_out = self.acoustic_tokenizer.encode(acoustic_input)
                if use_sample:
                    acoustic_tokens = acoustic_out.sample(dist_type=self.acoustic_tokenizer.std_dist_type)[0]
                else:
                    acoustic_tokens = acoustic_out.mean

                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                semantic_out = self.semantic_tokenizer.encode(acoustic_input)
                semantic_tokens = semantic_out.mean
                semantic_embeds = self.semantic_connector(semantic_tokens)
            else:
                # ==========================================
                # Streaming path (Retained for future use)
                # ==========================================
                acoustic_cache = VibeVoiceTokenizerStreamingCache()
                semantic_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_mean_segments = []
                semantic_mean_segments = []
                batch_size = audio.shape[0]
                sample_indices = torch.arange(batch_size, device=audio.device)

                def _iter_segments(total_length: int, segment_length: int):
                    for start in range(0, total_length, segment_length):
                        end = min(start + segment_length, total_length)
                        if end > start:
                            yield start, end

                segments = list(_iter_segments(total_samples, segment_samples))
                num_segments = len(segments)
                for seg_idx, (start, end) in enumerate(segments):
                    chunk = audio[:, start:end].contiguous()
                    if chunk.numel() == 0:
                        continue

                    # Check if this is the final segment
                    is_final = seg_idx == num_segments - 1

                    # --- Acoustic Encode ---
                    acoustic_enc_out = self.acoustic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=acoustic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    acoustic_mean_segments.append(acoustic_enc_out.mean)

                    # --- Semantic Encode ---
                    semantic_enc_out = self.semantic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=semantic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    semantic_mean_segments.append(semantic_enc_out.mean)

                if len(acoustic_mean_segments) == 0:
                    acoustic_mean_full = torch.zeros(
                        (batch_size, 0, self.acoustic_vae_dim),
                        device=audio.device,
                        dtype=self._audio_encoder_dtype,
                    )
                else:
                    acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1).contiguous()

                acoustic_enc_full = VibeVoiceTokenizerEncoderOutput(
                    mean=acoustic_mean_full,
                    std=self.acoustic_tokenizer.fix_std,
                )
                if use_sample:
                    acoustic_tokens = acoustic_enc_full.sample(dist_type=self.acoustic_tokenizer.std_dist_type)[0]
                else:
                    acoustic_tokens = acoustic_enc_full.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                # Concatenate sequence outputs (Semantic)
                if len(semantic_mean_segments) == 0:
                    semantic_tokens = torch.zeros(
                        (batch_size, 0, self.semantic_vae_dim),
                        device=audio.device,
                        dtype=self._audio_encoder_dtype,  # Use config dtype
                    )
                else:
                    semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
                # Connector uses same dtype as tokenizer
                semantic_embeds = self.semantic_connector(semantic_tokens)

        # Combine acoustic and semantic embeddings
        combined_embeds = acoustic_embeds + semantic_embeds

        # Convert to language model dtype for compatibility
        # Audio encoder uses config.torch_dtype (typically float32) for numerical precision,
        # but LM expects the dtype specified by vLLM's --dtype flag (e.g., bfloat16, float16)
        combined_embeds = combined_embeds.to(dtype=self._lm_dtype)

        return combined_embeds
    
class VibeVoiceASRProcessingInfo(BaseProcessingInfo):
    """Processing info for VibeVoice ASR multimodal model."""

    @staticmethod
    def load_audio(audio_path: str, target_sr: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
        """Load and normalize audio from file path."""
        audio, _ = load_audio_use_ffmpeg(audio_path, resample=True, target_sr=target_sr)
        audio = AudioNormalizer()(audio)
        return audio

    @staticmethod
    def audio_input_mapper(ctx, data: str | bytes | np.ndarray | list[str]) -> MultiModalInputs:
        """Map audio input data to vLLM MultiModalInputs format."""
        if isinstance(data, list):
            data = data[0]

        if isinstance(data, str):
            audio_waveform = VibeVoiceASRProcessingInfo.load_audio(data)
        elif isinstance(data, bytes):
            audio_waveform, _ = load_audio_bytes_use_ffmpeg(data, resample=True, target_sr=AUDIO_SAMPLE_RATE)
            audio_waveform = AudioNormalizer()(audio_waveform)
        elif isinstance(data, np.ndarray):
            audio_waveform = data
        else:
            raise ValueError(f"Unsupported audio data type: {type(data)}")

        audio_tensor = torch.from_numpy(audio_waveform).float()

        return MultiModalInputs(
            {
                "audio": audio_tensor,
                "audio_length": audio_tensor.shape[0],
            }
        )

    @staticmethod
    def get_field_config(hf_inputs: Mapping[str, torch.Tensor]):
        """Map HF processor output keys to audio modality."""
        config = {
            "raw_audio": MultiModalFieldConfig.batched("audio"),
            "raw_audio_lengths": MultiModalFieldConfig.batched("audio"),
            "salt": MultiModalFieldConfig.batched("audio"),
        }

        if "input_features" in hf_inputs:
            config["input_features"] = MultiModalFieldConfig.batched("audio")
        if "feature_attention_mask" in hf_inputs:
            config["feature_attention_mask"] = MultiModalFieldConfig.batched("audio")

        return config

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=AUDIO_SAMPLE_RATE)

    def get_feature_extractor(self, **kwargs) -> WhisperFeatureExtractor:
        model_path = self.ctx.model_config.model
        preprocessor_path = os.path.join(model_path, "preprocessor_config.json")

        config = {
            "sampling_rate": AUDIO_SAMPLE_RATE,
            "feature_size": 128,
            "hop_length": 240,
            "chunk_length": 30,
            "n_fft": 400,
            "padding_value": 0.0,
        }

        if os.path.exists(preprocessor_path):
            try:
                with open(preprocessor_path) as f:
                    file_config = json.load(f)
                    config.update({k: file_config[k] for k in config if k in file_config})
            except Exception as e:
                logger.warning(
                    f"Failed to load preprocessor config from {preprocessor_path}: {str(e)}. "
                    "Using default feature extractor configuration."
                )

        return WhisperFeatureExtractor(**config)

    def get_audio_token_info(self) -> dict[str, Any]:
        """Get audio special tokens and their IDs."""
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.get_vocab()

        tokens = {
            "audio_token": "<|AUDIO|>",
            "audio_bos_token": "<|audio_bos|>",
            "audio_eos_token": "<|audio_eos|>",
        }

        tokens["audio_token_id"] = vocab.get(tokens["audio_token"])
        tokens["audio_bos_id"] = vocab.get(tokens["audio_bos_token"])
        tokens["audio_eos_id"] = vocab.get(tokens["audio_eos_token"])

        return tokens

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class VibeVoiceASRDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceASRProcessingInfo]):
    """Dummy inputs builder for VibeVoice ASR model profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Generate dummy text with audio placeholders."""
        num_audios = mm_counts.get("audio", 0)
        if num_audios <= 0:
            return ""

        token_info = self.info.get_audio_token_info()
        return token_info["audio_token"] * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate dummy audio data for profiling."""
        feature_extractor = self.info.get_feature_extractor()
        audio_len = feature_extractor.chunk_length * feature_extractor.sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {"audio": [np.zeros(audio_len, dtype=np.float32) for _ in range(num_audios)]}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> ProcessorInputs:
        """Build ProcessorInputs for dummy profiling."""
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data)

        return ProcessorInputs(
            prompt=self.get_dummy_text(mm_counts),
            mm_items=dummy_mm_items,
        )


class VibeVoiceASRMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceASRProcessingInfo]):
    """Multi-modal processor for VibeVoice ASR model."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", None)
        if audios is not None and "audio" not in mm_data:
            mm_data["audio"] = audios

        if "audio" not in mm_data or mm_data["audio"] is None:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        raw_audio_list = mm_data.get("audio")
        if isinstance(raw_audio_list, np.ndarray):
            raw_audio_list = [raw_audio_list]
        elif not isinstance(raw_audio_list, list):
            raw_audio_list = list(raw_audio_list)

        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)

        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        max_len = max(len(a) for a in raw_audio_list)
        raw_audio_tensors = []
        audio_lengths = []
        for audio in raw_audio_list:
            audio_len = len(audio)
            audio_lengths.append(audio_len)
            if audio_len < max_len:
                audio = np.pad(audio, (0, max_len - audio_len), mode="constant")
            raw_audio_tensors.append(torch.from_numpy(audio).float())

        result["raw_audio"] = torch.stack(raw_audio_tensors, dim=0)
        result["raw_audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)

        salt_val = hash(str(uuid.uuid4())) % 100000
        result["salt"] = torch.tensor([salt_val], dtype=torch.long).expand(len(raw_audio_list))

        return result

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """Return whether the HF processor applies prompt updates."""
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Configure which HF output fields map to which modality."""
        return VibeVoiceASRProcessingInfo.get_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        token_info = self.info.get_audio_token_info()
        audio_token = token_info["audio_token"]
        audio_token_id = token_info["audio_token_id"]
        audio_bos_id = token_info.get("audio_bos_id")
        audio_eos_id = token_info.get("audio_eos_id")

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        def _tok_id(name: str) -> int | None:
            return vocab.get(name)

        speech_start_id = (
            _tok_id("<|object_ref_start|>")
            or getattr(tokenizer, "speech_start_id", None)
            or _tok_id("<|speech_start|>")
        )
        speech_end_id = (
            _tok_id("<|object_ref_end|>") or getattr(tokenizer, "speech_end_id", None) or _tok_id("<|speech_end|>")
        )
        speech_pad_id = (
            _tok_id("<|box_start|>") or getattr(tokenizer, "speech_pad_id", None) or _tok_id("<|speech_pad|>")
        )

        if audio_token_id is None:
            return []

        out_mm_data = out_mm_kwargs.get_data()
        raw_audio_lengths = out_mm_data.get("raw_audio_lengths", [])

        hf_config = self.info.get_hf_config()
        if isinstance(hf_config, dict):
            compress_ratio = int(hf_config.get("speech_tok_compress_ratio", 3200))
        else:
            compress_ratio = int(getattr(hf_config, "speech_tok_compress_ratio", 3200))

        def _to_int_len(x) -> int:
            if x is None:
                return 0
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return int(x.item())
                return int(x.shape[0])
            return int(x)

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            if raw_audio_lengths and item_idx < len(raw_audio_lengths):
                audio_len = _to_int_len(raw_audio_lengths[item_idx])
                num_features = max(1, int(np.ceil(audio_len / compress_ratio)))
            else:
                num_features = int(np.ceil(30 * AUDIO_SAMPLE_RATE / compress_ratio))

            if num_features == 0:
                raise ValueError(f"Audio at index {item_idx} is too short")

            # Derive the newline token ID from the tokenizer where possible, with a fallback.
            newline_id = _tok_id("\n") or getattr(tokenizer, "newline_id", None)
            if newline_id is None:
                try:
                    encoded_newline = tokenizer.encode("\n", add_special_tokens=False)
                    if isinstance(encoded_newline, list) and encoded_newline:
                        newline_id = int(encoded_newline[0])
                except Exception:
                    newline_id = None
            if newline_id is None:
                # Fallback to the legacy hardcoded value to preserve existing behavior.
                newline_id = 198
                logger.debug(f"Using fallback newline_id={newline_id} for tokenizer {type(tokenizer).__name__}")

            if speech_start_id is not None and speech_pad_id is not None and speech_end_id is not None:
                embed_id = int(speech_pad_id)
                replacement_ids = [int(speech_start_id)] + [embed_id] * num_features + [int(speech_end_id), newline_id]
            elif audio_bos_id is not None and audio_eos_id is not None:
                embed_id = int(audio_token_id)
                replacement_ids = [int(audio_bos_id)] + [embed_id] * num_features + [int(audio_eos_id)]
            else:
                embed_id = int(audio_token_id)
                replacement_ids = [embed_id] * num_features

            return PromptUpdateDetails.select_token_id(replacement_ids, embed_token_id=int(embed_id))

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement,
            )
        ]
    
@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceASRMultiModalProcessor,
    info=VibeVoiceASRProcessingInfo,
    dummy_inputs=VibeVoiceASRDummyInputsBuilder,
)
class VibeVoiceASRForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    This model combines VibeVoice acoustic/semantic tokenizers for audio encoding
    with a causal language model for text generation.
    """

    supports_transcription: ClassVar[Literal[True]] = True
    supports_transcription_only: ClassVar[bool] = False
    supports_segment_timestamp: ClassVar[bool] = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.has_preprocess = False
        self.have_multimodal_outputs = False

        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._model_path = vllm_config.model_config.model

        self.audio_encoder = VibeVoiceAudioEncoder(config)

        decoder_config = getattr(config, "decoder_config", config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=decoder_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        lm_dtype = vllm_config.model_config.dtype
        if lm_dtype is not None:
            self.audio_encoder._lm_dtype = lm_dtype

        try:
            self.audio_encoder._ensure_audio_encoder_dtype()
        except Exception:
            pass

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass for VibeVoice ASR model."""
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if intermediate_tensors is not None:
            inputs_embeds = None

        language_model = self.language_model
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model

        hidden_states = language_model.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if hidden_states is None:
            return None

        return self.language_model.compute_logits(hidden_states)

    def get_input_embeddings(self) -> torch.nn.Module:
        """Return the text embedding layer."""
        if hasattr(self.language_model, "model") and hasattr(self.language_model.model, "embed_tokens"):
            return self.language_model.model.embed_tokens
        elif hasattr(self.language_model, "embed_tokens"):
            return self.language_model.embed_tokens

        inner = self.language_model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        if hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
            return inner.model.embed_tokens

        raise AttributeError("Cannot find embed_tokens layer")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: Literal["transcribe", "translate"]
    ) -> SpeechToTextConfig:
        """Get the speech to text config for the ASR model."""
        return SpeechToTextConfig(
            language=None,  # Auto-detect or use request language
            task_type=task_type,
        )
    
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: torch.Tensor | list[torch.Tensor] | None = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply token embeddings to input_ids and merge with multimodal embeddings."""
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        embed_tokens = self.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)

        if multimodal_embeddings is not None and is_multimodal is not None:
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds,
                multimodal_embeddings,
                is_multimodal,
            )

        return inputs_embeds

    def embed_multimodal(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        """Extract audio embeddings using VibeVoice's acoustic/semantic tokenizers."""
        raw_audio = kwargs.get("raw_audio")
        if raw_audio is None:
            raw_audio = kwargs.get("audio")
        raw_audio_lengths = kwargs.get("raw_audio_lengths")
        if raw_audio_lengths is None:
            raw_audio_lengths = kwargs.get("audio_length")

        if raw_audio is None:
            return ()

        if isinstance(raw_audio, (list, tuple)) and len(raw_audio) == 0:
            return ()

        def flatten_lengths(lengths):
            """Flatten nested lists/tensors of lengths to a single list."""
            if lengths is None:
                return []

            result = []
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.tolist()

            if isinstance(lengths, (list, tuple)):
                for item in lengths:
                    if isinstance(item, (list, tuple)):
                        result.extend(flatten_lengths(item))
                    elif isinstance(item, torch.Tensor):
                        if item.dim() == 0:
                            result.append(item.item())
                        else:
                            result.extend(item.tolist())
                    else:
                        result.append(item)
            else:
                result.append(lengths)
            return result

        raw_audio_lengths = flatten_lengths(raw_audio_lengths)

        embeddings = []

        if isinstance(raw_audio, torch.Tensor):
            if raw_audio.dim() == 3:
                num_audios = raw_audio.shape[0]
                audio_list = [raw_audio[i].squeeze(0) for i in range(num_audios)]
            elif raw_audio.dim() == 2:
                num_audios = raw_audio.shape[0]
                audio_list = [raw_audio[i] for i in range(num_audios)]
            else:
                audio_list = [raw_audio]
        else:
            audio_list = list(raw_audio)

        for i, audio_tensor in enumerate(audio_list):
            try:
                if isinstance(audio_tensor, list):
                    audio_tensor = torch.stack(audio_tensor)

                if not isinstance(audio_tensor, torch.Tensor):
                    audio_tensor = torch.tensor(audio_tensor)

                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                device = next(self.audio_encoder.parameters()).device
                audio_tensor = audio_tensor.to(device=device, dtype=torch.float32)

                if raw_audio_lengths and i < len(raw_audio_lengths):
                    actual_len = int(raw_audio_lengths[i])
                    if actual_len > 0 and actual_len <= audio_tensor.shape[-1]:
                        audio_tensor = audio_tensor[..., :actual_len]

                if audio_tensor.numel() < 160:
                    continue

                audio_embeds = self.audio_encoder(audio_tensor)
                final_embed = audio_embeds.squeeze(0)
                embeddings.append(final_embed)

            except Exception as e:
                logger.warning(f"[VibeVoice] Failed to encode audio at index {i}: {e}")

        return tuple(embeddings)

    def get_language_model(self) -> torch.nn.Module:
        """Return the language model backbone."""
        return self.language_model

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights from checkpoint."""
        from vllm.model_executor.models.utils import (
            AutoWeightsLoader,
            WeightsMapper,
        )

        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.acoustic_tokenizer.": "audio_encoder.acoustic_tokenizer.",
                "model.semantic_tokenizer.": "audio_encoder.semantic_tokenizer.",
                "model.acoustic_connector.": "audio_encoder.acoustic_connector.",
                "model.semantic_connector.": "audio_encoder.semantic_connector.",
                "model.language_model.": "language_model.model.",
                "lm_head.": "language_model.lm_head.",
            }
        )

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Return the placeholder string format for a given modality."""
        if modality.startswith("audio"):
            return "<|AUDIO|>"
        raise ValueError("Only audio modality is supported")


__all__ = [
    "VibeVoiceASRForConditionalGeneration",
]