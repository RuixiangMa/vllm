import os
import json
import math
import warnings
import threading
from functools import partial
from subprocess import run
from dataclasses import dataclass
import copy
from typing import List, Optional, Union, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers import AutoProcessor
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.utils import TensorType, logging
from transformers.activations import ACT2FN

HAS_FFMPEG_UTILS = True

logger = logging.get_logger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."

COMMON_AUDIO_EXTS = [
    '.mp3', '.MP3', '.Mp3',
    '.m4a',
    '.mp4', '.MP4',
    '.wav', '.WAV',
    '.m4v',
    '.aac',
    '.ogg',
    '.mov', '.MOV',
    '.opus',
    '.m4b',
    '.flac',
    '.wma', '.WMA',
    '.rm', '.3gp', '.mpeg', '.flv', '.webm', '.mp2', '.aif', '.aiff', '.oga', '.ogv', '.mpga', '.m3u8', '.amr'
]

AUDIO_SAMPLE_RATE = 24000

from vllm.transformers_utils.configs.vibevoice_asr import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceSemanticTokenizerConfig,
)

def _get_ffmpeg_max_concurrency() -> int:
    """Get the maximum FFmpeg concurrency from environment variable."""
    v = os.getenv("VIBEVOICE_FFMPEG_MAX_CONCURRENCY", "")
    try:
        n = int(v) if v.strip() else 0
    except Exception:
        n = 0
    return n


_FFMPEG_MAX_CONCURRENCY = _get_ffmpeg_max_concurrency()
_FFMPEG_SEM = threading.Semaphore(_FFMPEG_MAX_CONCURRENCY) if _FFMPEG_MAX_CONCURRENCY > 0 else None


def _run_ffmpeg(cmd: list, *, stdin_bytes: bytes = None):
    """Run ffmpeg with optional global concurrency limiting.

    This is important for vLLM multi-request concurrency: spawning too many
    ffmpeg processes can saturate CPU/IO and cause request failures/timeouts.
    """
    if _FFMPEG_SEM is None:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)
    with _FFMPEG_SEM:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)


def load_audio_use_ffmpeg(file: str, resample: bool = False, target_sr: int = 24000):
    """Open an audio file and read as mono waveform, optionally resampling.

    Returns both the audio data and the original sample rate.
    """
    if not resample:
        cmd_probe = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file
        ]
        original_sr = int(run(cmd_probe, capture_output=True, check=True).stdout.decode().strip())
    else:
        original_sr = None

    sr_to_use = target_sr if resample else original_sr

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr_to_use),
        "-",
    ]

    out = _run_ffmpeg(cmd).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    return audio_data, sr_to_use


def load_audio_bytes_use_ffmpeg(data: bytes, *, resample: bool = False, target_sr: int = 24000):
    """Decode audio bytes via ffmpeg stdin pipe."""
    if not resample:
        raise ValueError("load_audio_bytes_use_ffmpeg requires resample=True")

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-",
    ]
    out = _run_ffmpeg(cmd, stdin_bytes=data).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio_data, target_sr


class AudioNormalizer:
    """Audio normalization class for VibeVoice tokenizer."""

    def __init__(self, target_dB_FS: float = -25, eps: float = 1e-6):
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def tailor_dB_FS(self, audio: np.ndarray) -> tuple:
        """Adjust the audio to the target dB FS level."""
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        normalized_audio = audio * scalar
        return normalized_audio, rms, scalar

    def avoid_clipping(self, audio: np.ndarray, scalar: Optional[float] = None) -> tuple:
        """Avoid clipping by scaling down if necessary."""
        if scalar is None:
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                scalar = max_val + self.eps
            else:
                scalar = 1.0
        return audio / scalar, scalar

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Normalize the audio by adjusting to target dB FS and avoiding clipping."""
        audio, _, _ = self.tailor_dB_FS(audio)
        audio, _ = self.avoid_clipping(audio)
        return audio


class VibeVoiceASRProcessor(ProcessorMixin):
    """
    Processor for VibeVoice ASR (Automatic Speech Recognition) models.
    
    This processor handles audio preprocessing and tokenization for ASR tasks,
    following the exact format used in training with proper chat templates.
    
    Args:
        tokenizer: The text tokenizer for processing text
        audio_processor: The audio processor for processing speech
        speech_tok_compress_ratio (int): Compression ratio for speech tokenization
        target_sample_rate (int): Target sample rate for audio
        normalize_audio (bool): Whether to normalize audio input
    """

    attributes = ["tokenizer", "audio_processor"]
    audio_processor_class = "VibeVoiceTokenizerProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        tokenizer=None,
        audio_processor=None,
        speech_tok_compress_ratio=320,
        target_sample_rate=24000,
        normalize_audio=True,
        **kwargs
    ):
        super().__init__(tokenizer, audio_processor)
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.target_sample_rate = target_sample_rate
        self.normalize_audio = normalize_audio

        if normalize_audio:
            self.audio_normalizer = AudioNormalizer()
        else:
            self.audio_normalizer = None

        self._cache_special_tokens()
    
    def _cache_special_tokens(self):
        """Cache special token IDs for efficiency."""
        # Add safety checks for special tokens
        if hasattr(self.tokenizer, 'speech_start_id'):
            self.speech_start_id = self.tokenizer.speech_start_id
        else:
            self.speech_start_id = self.tokenizer.convert_tokens_to_ids("<|speech_start|>")
            
        if hasattr(self.tokenizer, 'speech_end_id'):
            self.speech_end_id = self.tokenizer.speech_end_id
        else:
            self.speech_end_id = self.tokenizer.convert_tokens_to_ids("<|speech_end|>")
            
        if hasattr(self.tokenizer, 'speech_pad_id'):
            self.speech_pad_id = self.tokenizer.speech_pad_id
        else:
            self.speech_pad_id = self.tokenizer.convert_tokens_to_ids("<|speech_pad|>")
            
        if hasattr(self.tokenizer, 'pad_id'):
            self.pad_id = self.tokenizer.pad_id
        elif hasattr(self.tokenizer, 'pad_token_id'):
            self.pad_id = self.tokenizer.pad_token_id
        else:
            self.pad_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from a pretrained model path.
        
        Args:
            pretrained_model_name_or_path: Path to the pretrained model
            **kwargs: Additional keyword arguments
            
        Returns:
            VibeVoiceASRProcessor: The loaded processor
        """
        import json
        from transformers.utils import cached_file
        
        # Try to load configuration
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        config = {}
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path,
                    "preprocessor_config.json",
                    **kwargs
                )
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load preprocessor_config.json: {e}")
                logger.warning("Using default configuration")
        
        # Extract parameters
        speech_tok_compress_ratio = config.get("speech_tok_compress_ratio", 3200)
        target_sample_rate = config.get("target_sample_rate", 24000)
        normalize_audio = config.get("normalize_audio", True)
        
        # Load tokenizer
        language_model_pretrained_name = config.get("language_model_pretrained_name", None) or kwargs.pop("language_model_pretrained_name", "Qwen/Qwen2.5-1.5B")
        logger.info(f"Loading tokenizer from {language_model_pretrained_name}")
        
        if 'qwen' in language_model_pretrained_name.lower():
            tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(
                language_model_pretrained_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported tokenizer type for {language_model_pretrained_name}")
        
        # Load audio processor
        audio_processor = VibeVoiceTokenizerProcessor(
            sampling_rate=target_sample_rate,
            normalize_audio=normalize_audio,
            target_dB_FS=config.get("target_dB_FS", -25),
            eps=config.get("eps", 1e-6),
        )
        
        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=speech_tok_compress_ratio,
            target_sample_rate=target_sample_rate,
            normalize_audio=normalize_audio,
        )
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save processor configuration to a directory.
        
        Args:
            save_directory: Directory to save the configuration
            **kwargs: Additional keyword arguments
        """
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save processor configuration
        processor_config = {
            "processor_class": "VibeVoiceASRProcessor",
            "speech_tok_compress_ratio": self.speech_tok_compress_ratio,
            "target_sample_rate": self.target_sample_rate,
            "normalize_audio": self.normalize_audio,
            "target_dB_FS": -25,
            "eps": 1e-6,
        }
        
        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)
        
        logger.info(f"Processor configuration saved in {config_path}")
    
    def __call__(
        self,
        audio: Optional[Union[str, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        add_generation_prompt: bool = True,
        use_streaming: bool = True,
        context_info: Optional[str] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        Process audio input for ASR model.
        
        Args:
            audio: Audio input(s). Can be:
                - str: Path to audio file
                - np.ndarray: Audio array
                - torch.Tensor: Audio tensor
                - List of the above for batch processing
            sampling_rate: Sampling rate of input audio
            return_tensors: Output format ('pt' for PyTorch, 'np' for NumPy)
            padding: Whether to pad batch inputs
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            add_generation_prompt: Whether to add generation prompt for inference
            use_streaming: Whether to use streaming mode (True by default, auto False if <60s)
            context_info: Optional context information (e.g., hotwords, metadata) to help transcription
            
        Returns:
            BatchEncoding with:
                - input_ids: Token IDs for the model
                - attention_mask: Attention mask
                - acoustic_input_mask: Mask indicating speech token positions
                - speech_tensors: Processed speech features
                - speech_masks: Valid speech masks
                - vae_tok_seqlens: Length of each speech segment in tokens
        """
        if audio is None:
            raise ValueError("Audio input is required for ASR processing")
        
        # Handle single vs batch input
        if isinstance(audio, list):
            is_batched = True
            audio_list = audio
        else:
            is_batched = False
            audio_list = [audio]
        
        # Process each audio input
        all_encodings = []
        for audio_input in audio_list:
            encoding = self._process_single_audio(
                audio_input,
                sampling_rate=sampling_rate,
                add_generation_prompt=add_generation_prompt,
                use_streaming=use_streaming,
                context_info=context_info,
            )
            all_encodings.append(encoding)
        
        # Combine into batch
        batch_encoding = self._batch_encode(
            all_encodings,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        
        return batch_encoding
    
    def _process_single_audio(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sampling_rate: Optional[int] = None,
        add_generation_prompt: bool = True,
        use_streaming: bool = True,
        context_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single audio input.
        
        Args:
            audio: Single audio input
            sampling_rate: Audio sampling rate
            add_generation_prompt: Whether to add generation prompt
            context_info: Optional context information (e.g., hotwords, metadata) to help transcription
            
        Returns:
            Dictionary with processed tokens and audio features
        """
        # Process audio through audio processor
        if isinstance(audio, str):
            # Load from file using ffmpeg for better format support
            if HAS_FFMPEG_UTILS:
                try:
                    audio_array, file_sr = load_audio_use_ffmpeg(audio, resample=False)
                except Exception as e:
                    # Fall back to soundfile if ffmpeg fails
                    warnings.warn(f"ffmpeg loading failed, falling back to soundfile: {e}")
                    import soundfile as sf
                    audio_array, file_sr = sf.read(audio)
                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)  # Convert to mono
            else:
                import soundfile as sf
                audio_array, file_sr = sf.read(audio)
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)  # Convert to mono
            
            # Resample if needed
            if file_sr != self.target_sample_rate:
                import librosa
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=file_sr, 
                    target_sr=self.target_sample_rate
                )
        elif isinstance(audio, torch.Tensor):
            audio_array = audio.cpu().numpy()
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()
        else:
            audio_array = np.array(audio, dtype=np.float32)
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()
        
        # Ensure float32
        audio_array = audio_array.astype(np.float32)
        
        # Normalize if needed
        if self.normalize_audio and self.audio_normalizer:
            audio_array = self.audio_normalizer(audio_array)
        
        # Calculate audio duration
        audio_duration = len(audio_array) / self.target_sample_rate
        
        # Auto-disable streaming for short audio (<60s)
        if use_streaming and audio_duration < 60.0:
            use_streaming = False
        
        # Calculate token length based on streaming mode
        # Non-streaming: uses ceil (encoder adds extra_padding for stride alignment)
        # Streaming: uses floor (segments processed independently, no global alignment)
        # if use_streaming:
        #     vae_tok_len = len(audio_array) // self.speech_tok_compress_ratio
        # else:
        vae_tok_len = math.ceil(len(audio_array) / self.speech_tok_compress_ratio)
        
        # Build token sequence following training format
        # 1. System prompt - use apply_chat_template then encode like in training
        system_prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}],
            tokenize=False
        )
        system_tokens = self.tokenizer.encode(system_prompt_text)
        
        # 2. User input with speech tokens
        # Build speech placeholder string
        sp_start_token = self.tokenizer.convert_ids_to_tokens(self.speech_start_id)
        sp_pad_token = self.tokenizer.convert_ids_to_tokens(self.speech_pad_id)
        sp_end_token = self.tokenizer.convert_ids_to_tokens(self.speech_end_id)
        
        # User suffix with audio duration info
        show_keys = ['Start time', 'End time', 'Speaker ID', 'Content']
        if context_info and context_info.strip():
            user_suffix = f"This is a {audio_duration:.2f} seconds audio, with extra info: {context_info.strip()}\n\nPlease transcribe it with these keys: " + ", ".join(show_keys)
        else:
            user_suffix = f"This is a {audio_duration:.2f} seconds audio, please transcribe it with these keys: " + ", ".join(show_keys)
        
        user_input_string = ''.join(
            [sp_start_token] + [sp_pad_token] * vae_tok_len + [sp_end_token]
        ) + '\n' + user_suffix
        
        user_tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input_string}],
            tokenize=True
        )
        
        # Combine tokens
        full_tokens = system_tokens + user_tokens
        
        # Create acoustic input mask
        acoustic_input_mask = [1 if token == self.speech_pad_id else 0 for token in full_tokens]
        
        return {
            "input_ids": full_tokens,
            "acoustic_input_mask": acoustic_input_mask,
            "speech": audio_array,
            "vae_tok_len": vae_tok_len,
        }
    
    def _batch_encode(
        self,
        encodings: List[Dict[str, Any]],
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding:
        """
        Combine multiple encodings into a batch.
        
        Args:
            encodings: List of encoded samples
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            truncation: Whether to truncate
            return_tensors: Output format
            
        Returns:
            BatchEncoding with batched data
        """
        # Extract components
        input_ids_list = [enc["input_ids"] for enc in encodings]
        acoustic_masks_list = [enc["acoustic_input_mask"] for enc in encodings]
        speech_list = [enc["speech"] for enc in encodings]
        vae_tok_lens = [enc["vae_tok_len"] for enc in encodings]
        
        # Determine max length for padding
        if padding:
            if max_length is not None:
                target_length = max_length
            else:
                target_length = max(len(ids) for ids in input_ids_list)
            
            # Pad sequences
            padded_input_ids = []
            padded_acoustic_masks = []
            attention_masks = []
            
            for input_ids, acoustic_mask in zip(input_ids_list, acoustic_masks_list):
                # Truncate if needed
                if truncation and len(input_ids) > target_length:
                    input_ids = input_ids[:target_length]
                    acoustic_mask = acoustic_mask[:target_length]
                
                # Pad sequences to left (for autoregressive generation)
                padding_length = target_length - len(input_ids)
                padded_ids = [self.pad_id] * padding_length + input_ids
                padded_acoustic = [0] * padding_length + acoustic_mask
                attention_mask = [0] * padding_length + [1] * len(input_ids)
                
                padded_input_ids.append(padded_ids)
                padded_acoustic_masks.append(padded_acoustic)
                attention_masks.append(attention_mask)
            
            input_ids_list = padded_input_ids
            acoustic_masks_list = padded_acoustic_masks
        else:
            attention_masks = [[1] * len(ids) for ids in input_ids_list]
        
        # Process speech tensors - raw audio is 1D, so we keep it as is
        max_speech_length = max(len(s) for s in speech_list)
        padded_speeches = np.zeros((len(speech_list), max_speech_length), dtype=np.float32)
        speech_masks = np.zeros((len(speech_list), max(vae_tok_lens)), dtype=bool)
        
        for i, (speech, vae_len) in enumerate(zip(speech_list, vae_tok_lens)):
            padded_speeches[i, :len(speech)] = speech
            speech_masks[i, :vae_len] = True
        
        # Create batch encoding
        batch_encoding = BatchEncoding()
        
        if return_tensors == "pt":
            batch_encoding["input_ids"] = torch.tensor(input_ids_list, dtype=torch.long)
            batch_encoding["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
            batch_encoding["acoustic_input_mask"] = torch.tensor(acoustic_masks_list, dtype=torch.bool)
            batch_encoding["speech_tensors"] = torch.tensor(padded_speeches, dtype=torch.float32)
            batch_encoding["speech_masks"] = torch.tensor(speech_masks, dtype=torch.bool)
            # Note: vae_tok_seqlens and speech_type are not included as they are not model inputs
        else:
            batch_encoding["input_ids"] = input_ids_list if len(input_ids_list) > 1 else input_ids_list[0]
            batch_encoding["attention_mask"] = attention_masks if len(attention_masks) > 1 else attention_masks[0]
            batch_encoding["acoustic_input_mask"] = acoustic_masks_list if len(acoustic_masks_list) > 1 else acoustic_masks_list[0]
            batch_encoding["speech_tensors"] = padded_speeches if len(padded_speeches) > 1 else padded_speeches[0]
            batch_encoding["speech_masks"] = speech_masks if len(speech_masks) > 1 else speech_masks[0]
        
        return batch_encoding
    
    def batch_decode(self, *args, **kwargs):
        """
        Decode batch of token IDs to text.
        Forwards to tokenizer's batch_decode method.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """
        Decode token IDs to text.
        Forwards to tokenizer's decode method.
        """
        return self.tokenizer.decode(*args, **kwargs)
    
    def post_process_transcription(self, text: str) -> List[Dict[str, Any]]:
        """
        Post-process the generated transcription text to extract structured data.
        
        Args:
            text: Generated text from the model
            
        Returns:
            List of dictionaries with transcription segments
        """
        try:
            # Try to parse as JSON
            if "```json" in text:
                # Extract JSON from markdown code block
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_str = text[json_start:json_end].strip()
            else:
                # Try to find JSON array or object
                json_start = text.find("[")
                if json_start == -1:
                    json_start = text.find("{")
                if json_start != -1:
                    # Find matching closing bracket
                    bracket_count = 0
                    json_end = json_start
                    for i in range(json_start, len(text)):
                        if text[i] in "[{":
                            bracket_count += 1
                        elif text[i] in "]}":
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_end = i + 1
                                break
                    json_str = text[json_start:json_end]
                else:
                    json_str = text
            
            # Parse JSON
            result = json.loads(json_str)
            
            # Ensure it's a list
            if isinstance(result, dict):
                result = [result]
            
            # Validate and clean up the result
            cleaned_result = []
            for item in result:
                if isinstance(item, dict):
                    cleaned_item = {}
                    # Map keys to expected format
                    key_mapping = {
                        "Start time": "start_time",
                        "Start": "start_time",
                        "End time": "end_time",
                        "End": "end_time",
                        "Speaker ID": "speaker_id",
                        "Speaker": "speaker_id",
                        "Content": "text",
                    }
                    for key, mapped_key in key_mapping.items():
                        if key in item:
                            cleaned_item[mapped_key] = item[key]
                    
                    if cleaned_item:
                        cleaned_result.append(cleaned_item)
            
            return cleaned_result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from transcription: {e}")
            logger.debug(f"Raw text: {text}")
            return []
        except Exception as e:
            logger.warning(f"Error post-processing transcription: {e}")
            return []
    
    @property
    def model_input_names(self):
        """Return the list of inputs accepted by the model."""
        return ["input_ids", "attention_mask", "acoustic_input_mask", "speech_tensors", "speech_masks"]


class VibeVoiceTextTokenizer(Qwen2Tokenizer):
    """
    Construct a VibeVoice tokenizer. Based on the Qwen2 tokenizer with additional special tokens for speech.
    
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not used for vibevoice.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding.
        add_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether or not to add special tokens when encoding.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        add_special_tokens=True,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        
        # Add VibeVoice-specific special tokens
        self._add_vibevoice_special_tokens()
        
    def _add_vibevoice_special_tokens(self):
        """Add VibeVoice-specific special tokens."""
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",  # Speech start (reusing vision tokens)
                "<|vision_end|>",  # Speech end
                "<|vision_pad|>",  # Speech diffusion pad
            ]
        }
        num_added = self.add_special_tokens(special_tokens)
        
        # Cache special token IDs
        self._speech_start_id = self.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self.convert_tokens_to_ids("<|vision_pad|>")
        
        self._eos_id = self.convert_tokens_to_ids('<|endoftext|>')

        return num_added
    
    @property
    def eos_id(self) -> int:
        """Id of the end of sequence token."""
        return self._eos_id
    
    @property
    def speech_start_id(self) -> int:
        """Id of the speech start token."""
        return self._speech_start_id
    
    @property
    def speech_end_id(self) -> int:
        """Id of the speech end token."""
        return self._speech_end_id
    
    @property
    def speech_diffusion_id(self) -> int:
        """Id of the speech diffusion token."""
        return self._speech_diffusion_id
    
    @property
    def pad_id(self) -> int:
        """Id used for padding (returns -100 for loss masking)."""
        return -100


class VibeVoiceTextTokenizerFast(Qwen2TokenizerFast):
    """
    Construct a "fast" VibeVoice tokenizer (backed by HuggingFace's *tokenizers* library).
    Based on the Qwen2 tokenizer with additional special tokens for speech.
    
    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not used for vibevoice.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        
        # Add VibeVoice-specific special tokens
        self._add_vibevoice_special_tokens()
        
    def _add_vibevoice_special_tokens(self):
        """Add VibeVoice-specific special tokens."""
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",  # Speech start (reusing vision tokens)
                "<|vision_end|>",  # Speech end
                "<|vision_pad|>",  # Speech diffusion pad
            ]
        }
        num_added = self.add_special_tokens(special_tokens)
        
        # Cache special token IDs
        self._speech_start_id = self.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self.convert_tokens_to_ids("<|vision_pad|>")

        # self._eos_id = self.convert_tokens_to_ids('<|endoftext|>')
        self._eos_id = self.eos_token_id # qwen2 / qwen3
        self._pad_id = self.convert_tokens_to_ids('<|image_pad|>')
        
        return num_added
    
    @property
    def eos_id(self) -> int:
        """Id of the end of sequence token."""
        return self._eos_id
    
    @property
    def speech_start_id(self) -> int:
        """Id of the speech start token."""
        return self._speech_start_id
    
    @property
    def speech_end_id(self) -> int:
        """Id of the speech end token."""
        return self._speech_end_id
    
    @property
    def speech_diffusion_id(self) -> int:
        """Id of the speech diffusion token."""
        return self._speech_diffusion_id
    
    @property
    def pad_id(self) -> int:
        """Id used for padding (returns -100 for loss masking)."""
        return self._pad_id

class VibeVoiceASRTextTokenizerFast(Qwen2TokenizerFast):
    """
    Construct a "fast" VibeVoice tokenizer (backed by HuggingFace's *tokenizers* library).
    Based on the Qwen2 tokenizer with additional special tokens for speech.
    
    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not used for vibevoice.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        
        # Add VibeVoice-specific special tokens
        self._add_vibevoice_special_tokens()
        
        # https://github.com/QwenLM/Qwen2.5-VL/blob/d2240f11656bfe404b9ba56db4e51cd09f522ff1/qwen-vl-finetune/qwenvl/data/data_qwen_packed.py#L57C5-L57C222
        self.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        
    def _add_vibevoice_special_tokens(self):
        """Add VibeVoice-specific special tokens."""
        special_tokens = {
            "additional_special_tokens": [
                "<|object_ref_start|>",  # Speech start (reusing vision tokens)
                "<|object_ref_end|>",  # Speech end
                "<|box_start|>",  # Speech diffusion pad
            ]
        }
        num_added = self.add_special_tokens(special_tokens)
        
        # Cache special token IDs
        self._speech_start_id = self.convert_tokens_to_ids("<|object_ref_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|object_ref_end|>")
        self._speech_pad_id = self.convert_tokens_to_ids("<|box_start|>")

        self._eos_id = self.eos_token_id # qwen2 / qwen3
        self._pad_id = self.convert_tokens_to_ids('<|image_pad|>')
        
        return num_added    
    
    @property
    def eos_id(self) -> int:
        """Id of the end of sequence token."""
        return self._eos_id
    
    @property
    def speech_start_id(self) -> int:
        """Id of the speech start token."""
        return self._speech_start_id
    
    @property
    def speech_end_id(self) -> int:
        """Id of the speech end token."""
        return self._speech_end_id
    
    @property
    def speech_pad_id(self) -> int:
        """Id of the speech diffusion token."""
        return self._speech_pad_id
    
    @property
    def pad_id(self) -> int:
        return self._pad_id

class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        x = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(x) 
        x = x.transpose(1, 2)  # b t ... -> b ... t
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            weight_shape = (dim,) if weight_shape is None else weight_shape
            self.weight = nn.Parameter(torch.ones(weight_shape))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

class ConvRMSNorm(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__(dim, eps, elementwise_affine, weight_shape)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        output = output.transpose(1, 2)  # b t ... -> b ... t
        return output

# Convolutional layers and utilities
CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                'time_layer_norm', 'layer_norm', 'time_group_norm'])


def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return nn.utils.weight_norm(module)
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                padding_total: int = 0) -> int:
    """Calculate extra padding needed for convolution to have the same output length"""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor, paddings: Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Pad 1D input with handling for small inputs in reflect mode"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: Dict[str, Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: Dict[str, Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class VibeVoiceTokenizerStreamingCache:
    """Cache for streaming convolution, similar to KV cache in attention"""
    def __init__(self):
        self.cache = {}  # Dict mapping (layer_id, sample_idx) to state tensor
        
    def get(self, layer_id: str, sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached states for given layer and sample indices"""
        states = []
        max_length = 0
        
        # First pass: collect states and find max length
        for idx in sample_indices.tolist():
            key = (layer_id, idx)
            if key not in self.cache:
                return None  # If any sample is missing, return None
            state = self.cache[key]
            states.append(state)
            max_length = max(max_length, state.shape[-1])
        
        # Second pass: pad states to max length if needed
        if len(states) > 0 and states[0].dim() >= 2:
            padded_states = []
            for state in states:
                if state.shape[-1] < max_length:
                    # Pad on the time dimension (last dimension)
                    pad_size = max_length - state.shape[-1]
                    # Pad with zeros on the LEFT to align the most recent samples
                    padded_state = F.pad(state, (pad_size, 0), mode='constant', value=0)
                    padded_states.append(padded_state)
                else:
                    padded_states.append(state)
            return torch.stack(padded_states, dim=0)
        else:
            return torch.stack(states, dim=0)
    
    def set(self, layer_id: str, sample_indices: torch.Tensor, states: torch.Tensor):
        """Set cached states for given layer and sample indices"""
        for i, idx in enumerate(sample_indices.tolist()):
            key = (layer_id, idx)
            self.cache[key] = states[i].detach()

    def set_to_zero(self, sample_indices: torch.Tensor):
        """Set all cached states to zero for given sample indices"""
        for key in list(self.cache.keys()):
            layer_id, sample_idx = key
            if sample_idx in sample_indices.tolist():
                # Create zero tensor with same shape and dtype as cached tensor
                cached_tensor = self.cache[key]
                self.cache[key] = torch.zeros_like(cached_tensor)
                
    def clear(self, layer_id: Optional[str] = None, sample_indices: Optional[torch.Tensor] = None):
        """Clear cache for specific layer/samples or everything"""
        if layer_id is None and sample_indices is None:
            self.cache.clear()
        elif layer_id is not None and sample_indices is None:
            # Clear all samples for a specific layer
            keys_to_remove = [k for k in self.cache.keys() if k[0] == layer_id]
            for k in keys_to_remove:
                del self.cache[k]
        elif layer_id is not None and sample_indices is not None:
            # Clear specific samples for a specific layer
            for idx in sample_indices.tolist():
                key = (layer_id, idx)
                self.cache.pop(key, None)

class SConv1d(nn.Module):
    """Conv1d with built-in handling of asymmetric or causal padding and normalization."""
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, dilation: int = 1,
                groups: int = 1, bias: bool = True, causal: bool = False,
                norm: str = 'none', norm_kwargs: Dict[str, Any] = {},
                pad_mode: str = 'reflect'):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                            dilation=dilation, groups=groups, bias=bias, causal=causal,
                            norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode
        
        # Store configuration
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # For causal convolution, we need to maintain kernel_size - 1 samples as context
        # need to check use which context_size is more suitable
        # self.context_size = (kernel_size - 1) * dilation
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        
        # For non-streaming mode, calculate padding
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)
        
        # Create a unique layer ID for cache management
        self._layer_id = None
                  
    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconv1d_{id(self)}"
        return self._layer_id
        
    def forward(self, x: torch.Tensor, 
                cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
                sample_indices: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                debug: bool = False,
                is_final_chunk: bool = False) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        
        Args:
            x: Input tensor [batch_size, channels, time]
            cache: VibeVoiceTokenizerStreamingCache object for maintaining states
            sample_indices: Indices identifying each sample for cache management
            use_cache: Whether to use cached states for streaming
            debug: Whether to print debug information
            is_final_chunk: Whether this is the final chunk (adds extra padding for alignment)
            
        Returns:
            Output tensor
        """
        B, C, T = x.shape
        
        # Non-streaming mode
        if not use_cache or cache is None:
            return self._forward_non_streaming(x, debug=debug)
        
        # Streaming mode
        assert self.causal, "Streaming mode is only supported for causal convolutions"
        assert sample_indices is not None, "sample_indices must be provided for streaming mode"
        assert len(sample_indices) == B, "sample_indices must match batch size"
        
        return self._forward_streaming(x, cache, sample_indices, debug, is_final_chunk)
    
    def _forward_streaming(self, x: torch.Tensor, 
                          cache: VibeVoiceTokenizerStreamingCache,
                          sample_indices: torch.Tensor,
                          debug: bool = False,
                          is_final_chunk: bool = False) -> torch.Tensor:
        """Streaming forward pass with cache operations kept separate from compiled code"""
        B, C, T = x.shape
        
        # Cache operations (not compiled)
        cached_states = cache.get(self.layer_id, sample_indices)
        
        if cached_states is None:
            # First chunk - initialize with zeros for context
            if self.context_size > 0:
                cached_states = torch.zeros(B, C, self.context_size, device=x.device, dtype=x.dtype)
                if debug:
                    print(f"[DEBUG] Initialized cache with shape: {cached_states.shape}, context_size={self.context_size}")
            else:
                cached_states = torch.zeros(B, C, 0, device=x.device, dtype=x.dtype)
                if debug:
                    print(f"[DEBUG] No context needed (kernel_size=stride)")
        
        # Concatenate cached states with input
        if cached_states.shape[2] > 0:
            input_with_context = torch.cat([cached_states, x], dim=2)
        else:
            input_with_context = x
        
        # For final chunk, add extra padding to ensure ceil behavior (same as non-streaming)
        if is_final_chunk:
            extra_padding = get_extra_padding_for_conv1d(
                input_with_context, self.kernel_size, self.stride, self.padding_total
            )
            if extra_padding > 0:
                input_with_context = pad1d(input_with_context, (0, extra_padding), mode=self.pad_mode)
                if debug:
                    print(f"[DEBUG] Final chunk: added extra_padding={extra_padding}")
            
        if debug:
            print(f"[DEBUG] Input shape: {x.shape}, Cache shape: {cached_states.shape}, Combined: {input_with_context.shape}")
        
        # Apply convolution directly - no extra padding in streaming mode
        # The conv layer will handle its own padding internally
        output = self.conv(input_with_context)

        if debug:
            print(f"[DEBUG] Output shape: {output.shape}")
        
        # Update cache for next chunk
        if self.context_size > 0:
            # Calculate how many samples to keep
            total_input_length = input_with_context.shape[2]
            
            # Keep the last context_size samples
            if total_input_length >= self.context_size:
                new_cache_start = total_input_length - self.context_size
                new_cache = input_with_context[:, :, new_cache_start:]
            else:
                # If we have less than context_size samples, keep everything
                new_cache = input_with_context
                
            if debug:
                print(f"[DEBUG] New cache shape: {new_cache.shape}")
                
            cache.set(self.layer_id, sample_indices, new_cache)
        
        return output
    
    def _forward_non_streaming(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Standard forward pass without streaming"""
        B, C, T = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        padding_total = self.padding_total
        
        # Compute extra padding for stride alignment
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] Input shape: {x.shape}, padding_total={padding_total}, extra_padding={extra_padding}")
        
        if self.causal:
            # Left padding for causal
            if self.pad_mode == 'constant':
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode, value=0)
            else:
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Symmetric padding for non-causal
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] After padding: {x.shape}")
            
        output = self.conv(x)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] Output shape: {output.shape}")
        
        return output


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in handling of asymmetric or causal padding and normalization."""
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, causal: bool = False,
                norm: str = 'none', trim_right_ratio: float = 1.,
                norm_kwargs: Dict[str, Any] = {}, bias: bool = True):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                        causal=causal, norm=norm, norm_kwargs=norm_kwargs, bias=bias)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

        # Store configuration
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # For transposed convolution, padding calculation is different
        self.padding_total = kernel_size - stride
        
        # For streaming, we need to keep track of input history
        # Transposed conv needs to see multiple input samples to produce correct output
        self.context_size = kernel_size - 1
        
        # Create a unique layer ID for cache management
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconvtr1d_{id(self)}"
        return self._layer_id
    
    def forward(self, x: torch.Tensor,
                cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
                sample_indices: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                debug: bool = False) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        """
        B, C, T = x.shape
        
        # Non-streaming mode
        if not use_cache or cache is None:
            return self._forward_non_streaming(x, debug=debug)
        
        # Streaming mode
        assert sample_indices is not None, "sample_indices must be provided for streaming mode"
        assert len(sample_indices) == B, "sample_indices must match batch size"
        
        return self._forward_streaming(x, cache, sample_indices, debug)
    
    def _forward_streaming(self, x: torch.Tensor,
                          cache: VibeVoiceTokenizerStreamingCache,
                          sample_indices: torch.Tensor,
                          debug: bool = False) -> torch.Tensor:
        """Streaming forward pass with cache operations kept separate from compiled code"""
        B, C, T = x.shape
        
        # Cache operations (not compiled)
        cached_input = cache.get(self.layer_id, sample_indices)
        
        if cached_input is None:
            # First chunk - no history yet
            cached_input = torch.zeros(B, C, 0, device=x.device, dtype=x.dtype)
            if debug:
                print(f"[DEBUG] Initialized empty cache for transposed conv")
        
        # Concatenate cached input with new input
        full_input = torch.cat([cached_input, x], dim=2)
        
        if debug:
            print(f"[DEBUG] Input shape: {x.shape}, Cache shape: {cached_input.shape}, Combined: {full_input.shape}")
        
        # First chunk or debug mode - use uncompiled version
        full_output = self.convtr(full_input)
        
        if debug:
            print(f"[DEBUG] Full transposed conv output shape: {full_output.shape}")
        
        # Calculate padding to remove
        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
        
        # Remove padding
        if padding_left + padding_right > 0:
            full_output = unpad1d(full_output, (padding_left, padding_right))
        
        if debug:
            print(f"[DEBUG] After unpadding: {full_output.shape}")
        
        # Determine which part of the output corresponds to the new input
        if cached_input.shape[2] == 0:
            # First chunk - return all output
            output = full_output
        else:
            # Subsequent chunks - return only the new output
            expected_new_output = T * self.stride
            
            # Take the last expected_new_output samples
            if full_output.shape[2] >= expected_new_output:
                output = full_output[:, :, -expected_new_output:]
            else:
                output = full_output
        
        if debug:
            print(f"[DEBUG] Final streaming output shape: {output.shape}")
        
        # Update cache
        if full_input.shape[2] > self.context_size:
            new_cache = full_input[:, :, -self.context_size:]
        else:
            new_cache = full_input
        
        if debug:
            print(f"[DEBUG] New cache shape: {new_cache.shape}")
            
        cache.set(self.layer_id, sample_indices, new_cache)
        
        return output
    
    def _forward_non_streaming(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Standard forward pass without streaming"""
        if debug:
            print(f"[DEBUG NON-STREAMING] Input shape: {x.shape}")
        
        # Apply transposed convolution
        y = self.convtr(x)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] After transposed conv: {y.shape}")
        
        # Calculate and remove padding
        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
        
        if padding_left + padding_right > 0:
            y = unpad1d(y, (padding_left, padding_right))
        
        if debug:
            print(f"[DEBUG NON-STREAMING] Final output shape: {y.shape}")
            
        return y
    
# FFN 
class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        bias=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(self.embed_dim, ffn_dim, bias=bias) 
        self.gelu = ACT2FN["gelu"]
        self.linear2 = nn.Linear(ffn_dim, self.embed_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Convlayer(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            dilation=1, 
            groups=1, 
            bias=True, 
            pad_mode='zeros', 
            norm='weight_norm', 
            causal=True, 
        ):
        super().__init__()
        self.conv = SConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, 
                           groups=groups, bias=bias, pad_mode=pad_mode, norm=norm, causal=causal)

    def forward(self, x):
        return self.conv(x)

class Block1D(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0., mixer_layer='conv',  
                layer_scale_init_value=1e-6, **kwargs):
        super().__init__()
        
        if kwargs.get('layernorm', 'LN') == 'LN':
            self.norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))               
        elif kwargs.get('layernorm', 'RMSNorm') == 'RMSNorm':
            self.norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))

        if mixer_layer == 'conv':
            self.mixer = Convlayer(dim, dim, groups=kwargs.get('groups', 1),
                                kernel_size=kernel_size, 
                                pad_mode=kwargs.get('pad_mode', 'reflect'), 
                                norm=kwargs.get('norm', 'none'), 
                                causal=kwargs.get('causal', True), 
                                bias=kwargs.get('bias', True),
                                )
        elif mixer_layer == 'depthwise_conv':
            self.mixer = Convlayer(dim, dim, groups=dim,
                                kernel_size=kernel_size, 
                                pad_mode=kwargs.get('pad_mode', 'reflect'), 
                                norm=kwargs.get('norm', 'none'), 
                                causal=kwargs.get('causal', True), 
                                bias=kwargs.get('bias', True),
                                )
        else:
            raise ValueError(f"Unsupported mixer layer: {mixer_layer}")
        
        self.ffn = FFN(
            dim, 
            kwargs.get('ffn_expansion', 4) * dim, 
            bias=kwargs.get('bias', False),
        )
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.modules.DropPath(drop_path)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x):
        # mixer
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        # ffn
        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        return x


class TokenizerEncoder(nn.Module):
    """
    Encoder component for the VibeVoice tokenizer that converts audio to latent representations.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract parameters from config
        self.channels = config.channels
        self.dimension = config.dimension
        self.n_filters = config.n_filters
        self.ratios = list(reversed(config.ratios))
        self.depths = config.depths
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal
        
        # Additional config parameters with defaults
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        layernorm_elementwise_affine = getattr(config, "layernorm_elementwise_affine", True)
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)
        
        # determine the norm type based on layernorm
        if layernorm == 'LN':
            norm_type = ConvLayerNorm
        elif layernorm == 'RMSNorm':
            norm_type = partial(ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine)
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")
        
        # stem and intermediate downsampling conv layers
        stem = nn.Sequential(
                SConv1d(self.channels, self.n_filters, kernel_size, norm=norm, norm_kwargs=norm_params, causal=self.causal, pad_mode=pad_mode, bias=bias),
            )
        
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** i)
            out_ch = self.n_filters * (2 ** (i + 1))
            downsample_layer = nn.Sequential(
                SConv1d(in_ch, out_ch, kernel_size=self.ratios[i] * 2, stride=self.ratios[i], causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)
            )
            self.downsample_layers.append(downsample_layer)

        # configure the transformer blocks
        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0

        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** i)
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]
        
        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(in_ch, self.dimension, kernel_size=last_kernel_size, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)

    def forward_features(self, x, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        for i in range(len(self.depths)):
            # Apply downsampling
            for layer in self.downsample_layers[i]:
                if isinstance(layer, SConv1d):
                    x = layer(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
                else:
                    x = layer(x)
            
            # Apply stage (Block1D contains Convlayer which contains SConv1d)
            for block in self.stages[i]:
                if hasattr(block, 'mixer') and hasattr(block.mixer, 'conv') and isinstance(block.mixer.conv, SConv1d):
                    # Block1D forward with cache support
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x
                    
                    # FFN part
                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)

        return self.norm(x)

    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        x = self.forward_features(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        x = self.head(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        return x


class TokenizerDecoder(nn.Module):
    """
    Decoder component for the VibeVoice tokenizer that converts latent representations back to audio.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract parameters from config
        self.dimension = config.dimension
        self.channels = config.channels
        self.n_filters = config.n_filters
        self.ratios = config.ratios
        
        # IMPORTANT CHANGE: Don't reverse depths again since they're already reversed in VibeVoiceAcousticTokenizerModel
        self.depths = config.depths  # Changed from list(reversed(config.depths))
        
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal
        
        # Additional config parameters with defaults
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        trim_right_ratio = getattr(config, "trim_right_ratio", 1.0)
        layernorm_elementwise_affine = getattr(config, "layernorm_elementwise_affine", True)
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        # determine the norm type based on layernorm
        if layernorm == 'LN':
            norm_type = ConvLayerNorm
        elif layernorm == 'RMSNorm':
            norm_type = partial(ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine)
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")
        
        # stem and upsampling layers
        stem = nn.Sequential(
                SConv1d(self.dimension, self.n_filters * 2 ** (len(self.depths) - 1), kernel_size, norm=norm, 
                        norm_kwargs=norm_params, causal=self.causal, pad_mode=pad_mode, bias=bias),
            )
        
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            out_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i - 1))
            upsample_layer = nn.Sequential(
                SConvTranspose1d(in_ch, out_ch,
                                kernel_size=self.ratios[i] * 2, stride=self.ratios[i],
                                norm=norm, norm_kwargs=norm_params, bias=bias,
                                causal=self.causal, trim_right_ratio=trim_right_ratio),
            )
            self.upsample_layers.append(upsample_layer)

        # configure transformer blocks
        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0
        
        # Create stages in the same order as the original model
        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(in_ch, self.channels, kernel_size=last_kernel_size, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)

    def forward_features(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        for i in range(len(self.depths)):
            # Apply upsampling
            for layer in self.upsample_layers[i]:
                if isinstance(layer, (SConv1d, SConvTranspose1d)):
                    x = layer(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
                else:
                    x = layer(x)
            
            # Apply stage (Block1D contains Convlayer which contains SConv1d)
            for block in self.stages[i]:
                if hasattr(block, 'mixer') and hasattr(block.mixer, 'conv') and isinstance(block.mixer.conv, SConv1d):
                    # Block1D forward with cache support
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x
                    
                    # FFN part
                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)

        return self.norm(x)
    
    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        x = self.forward_features(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        x = self.head(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return x
    

@dataclass
class VibeVoiceTokenizerEncoderOutput:
    """
    Output of VibeVoice tokenizer encoder, representing a Gaussian distribution with fixed variance.
    
    Args:
        mean (`torch.FloatTensor`): The mean parameters of the distribution.
        std (`float` or `torch.FloatTensor`): Fixed standard deviation value.
    """
    mean: torch.Tensor
    std: Optional[Union[float, torch.Tensor]] = None
    
    def sample(self, dist_type='fix'):
        """
        Sample from the distribution.
        
        Args:
            dist_type (`str`): Sampling method, either 'fix' or 'gaussian'.
                
        Returns:
            `torch.FloatTensor`: Sampled values.
            `torch.FloatTensor` (optional): Standard deviation used (only when dist_type='gaussian').
        """
        if dist_type == 'fix':
            x = self.mean + self.std * torch.randn_like(self.mean)
            return x, self.std
        elif dist_type == 'gaussian':
            batch_size = self.mean.size(0)
            value = self.std / 0.8
            std = torch.randn(batch_size, device=self.mean.device, dtype=self.mean.dtype) * value

            while std.dim() < self.mean.dim():
                std = std.unsqueeze(-1)

            x = self.mean + std * torch.randn_like(self.mean)
            return x, std
        else:
            return self.mean, self.std

    def kl(self):
        """Compute KL divergence between this distribution and a standard normal."""
        target = torch.zeros_like(self.mean)
        return F.mse_loss(self.mean, target, reduction='none')

    def mode(self):
        """Return the distribution mode (which is the mean for Gaussian)."""
        return self.mean
    
class VibeVoiceAcousticTokenizerModel(PreTrainedModel):
    """VibeVoice speech tokenizer model combining encoder and decoder for acoustic tokens"""
    
    config_class = VibeVoiceAcousticTokenizerConfig
    base_model_prefix = "vibevoice_acoustic_tokenizer"
    _supports_flash_attn_2 = True  
    _supports_sdpa = True  
    _no_split_modules = ["TokenizerEncoder", "TokenizerDecoder"]

    def __init__(self, config):
        super().__init__(config)
        
        self.register_buffer('fix_std', torch.tensor(config.fix_std), persistent=False)
        self.std_dist_type = getattr(config, "std_dist_type", "fix")
        
        # Parse encoder depths
        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths
            
        # Parse decoder depths if provided
        if config.decoder_depths is not None and isinstance(config.decoder_depths, str):
            decoder_depths = [int(d) for d in config.decoder_depths.split('-')]
        else:
            # Default: use reversed encoder depths if decoder_depths is None
            decoder_depths = list(reversed(encoder_depths))
        
        # Create encoder config
        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm
        
        # Create decoder config
        decoder_config = copy.deepcopy(config)
        decoder_config.dimension = config.vae_dim
        decoder_config.n_filters = config.decoder_n_filters
        decoder_config.ratios = config.decoder_ratios
        decoder_config.depths = decoder_depths
        decoder_config.norm = config.conv_norm
        decoder_config.pad_mode = config.pad_mode
        decoder_config.bias = config.conv_bias
        decoder_config.layernorm_eps = config.layernorm_eps
        decoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        decoder_config.mixer_layer = config.mixer_layer
        decoder_config.layer_scale_init_value = config.layer_scale_init_value
        decoder_config.disable_last_norm = config.disable_last_norm
        
        # Initialize encoder and decoder
        self.encoder = TokenizerEncoder(encoder_config)
        self.decoder = TokenizerDecoder(decoder_config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the model"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @torch.no_grad()
    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        """Convert audio to latent representations"""
        latents = self.encoder(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1), std=self.fix_std)
    
    @torch.no_grad()
    def sampling(self, encoder_output, dist_type=None):
        """Sample from the encoder output distribution"""
        dist_type = dist_type or self.std_dist_type
    
        if dist_type == 'fix':
            return encoder_output.sample(dist_type='fix')
        elif dist_type == 'gaussian':
            return encoder_output.sample(dist_type='gaussian')
        else:
            raise ValueError(f"Unsupported dist_type: {dist_type}, expected 'fix' or 'gaussian'")
    
    @torch.no_grad()
    def decode(self, latents, cache=None, sample_indices=None, use_cache=False, debug=False):
        """Convert latent representations back to audio"""
        if latents.shape[1] == self.config.vae_dim:
            pass
        else:
            latents = latents.permute(0, 2, 1)

        audio = self.decoder(latents, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return audio

    def forward(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False):
        """Full forward pass: encode audio to latents, then decode back to audio"""
        encoder_output = self.encode(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        sampled_latents, _ = self.sampling(encoder_output)
        reconstructed = self.decode(sampled_latents, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return reconstructed, sampled_latents


class VibeVoiceSemanticTokenizerModel(PreTrainedModel):
    """VibeVoice speech tokenizer model with only encoder for semantic tokens"""
    
    config_class = VibeVoiceSemanticTokenizerConfig
    base_model_prefix = "vibevoice_semantic_tokenizer"
    _supports_flash_attn_2 = True  
    _supports_sdpa = True  
    _no_split_modules = ["TokenizerEncoder"]
    
    def __init__(self, config):
        super().__init__(config)
        
        # Parse encoder depths
        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths
        
        # Create encoder config
        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm
        
        # Initialize encoder and decoder
        self.encoder = TokenizerEncoder(encoder_config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the model"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @torch.no_grad()
    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False, is_final_chunk=False):
        """Convert audio to latent representations"""
        latents = self.encoder(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug, is_final_chunk=is_final_chunk)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1))
    
    @torch.no_grad()
    def sampling(self, encoder_output, dist_type=None):
        """Sample from the encoder output distribution"""
        return encoder_output.sample(dist_type='none')

    def forward(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False):
        """Full forward pass: encode audio to latents, then decode back to audio"""
        encoder_output = self.encode(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        sampled_latents, _ = self.sampling(encoder_output, dist_type='none')
        return None, sampled_latents
# Change from ProcessorMixin to FeatureExtractionMixin which is designed for single components
class VibeVoiceTokenizerProcessor(FeatureExtractionMixin):
    """
    Processor for VibeVoice acoustic tokenizer models.
    
    This processor handles audio preprocessing for VibeVoice models, including:
    - Audio format conversion (stereo to mono)
    - Optional audio normalization
    - Streaming support for infinite-length audio
    
    Args:
        sampling_rate (int, optional): Expected sampling rate. Defaults to 24000.
        normalize_audio (bool, optional): Whether to normalize audio. Defaults to True.
        target_dB_FS (float, optional): Target dB FS for normalization. Defaults to -25.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-6.
    """
    model_input_names = ["input_features"]
    
    def __init__(
        self,
        sampling_rate: int = 24000,
        normalize_audio: bool = True,
        target_dB_FS: float = -25,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        
        # Initialize audio normalizer if needed
        if self.normalize_audio:
            self.normalizer = AudioNormalizer(target_dB_FS=target_dB_FS, eps=eps)
        else:
            self.normalizer = None
        
        # Save config
        self.feature_extractor_dict = {
            "sampling_rate": sampling_rate,
            "normalize_audio": normalize_audio,
            "target_dB_FS": target_dB_FS,
            "eps": eps,
        }
    
    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono if needed.
        
        Args:
            audio (np.ndarray): Input audio array
            
        Returns:
            np.ndarray: Mono audio array
        """
        if len(audio.shape) == 1:
            return audio
        elif len(audio.shape) == 2:
            if audio.shape[0] == 2:  # (2, time)
                return np.mean(audio, axis=0)
            elif audio.shape[1] == 2:  # (time, 2)
                return np.mean(audio, axis=1)
            else:
                # If one dimension is 1, squeeze it
                if audio.shape[0] == 1:
                    return audio.squeeze(0)
                elif audio.shape[1] == 1:
                    return audio.squeeze(1)
                else:
                    raise ValueError(f"Unexpected audio shape: {audio.shape}")
        else:
            raise ValueError(f"Audio should be 1D or 2D, got shape: {audio.shape}")
    
    def _process_single_audio(self, audio: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Process a single audio array.
        
        Args:
            audio: Single audio input
            
        Returns:
            np.ndarray: Processed audio
        """
        # Convert to numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)
        
        # Ensure mono
        audio = self._ensure_mono(audio)
        
        # Normalize if requested
        if self.normalize_audio and self.normalizer is not None:
            audio = self.normalizer(audio)
        
        return audio
    
    def __call__(
        self,
        audio: Union[str, np.ndarray, List[float], List[np.ndarray], List[List[float]], List[str]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        """
        Process audio for VibeVoice models.
        
        Args:
            audio: Audio input(s) to process. Can be:
                - str: Path to audio file
                - np.ndarray: Audio array
                - List[float]: Audio as list of floats
                - List[np.ndarray]: Batch of audio arrays
                - List[str]: Batch of audio file paths
            sampling_rate (int, optional): Sampling rate of the input audio
            return_tensors (str, optional): Return format ('pt' for PyTorch, 'np' for NumPy)
            
        Returns:
            dict: Processed audio inputs with keys:
                - input_features: Audio tensor(s) ready for the model
        """
        if audio is None:
            raise ValueError("Audio input is required")
        
        # Validate sampling rate
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            logger.warning(
                f"Input sampling rate ({sampling_rate}) differs from expected "
                f"sampling rate ({self.sampling_rate}). Please resample your audio."
            )
        
        # Handle different input types
        if isinstance(audio, str):
            # Single audio file path
            audio = self._load_audio_from_path(audio)
            is_batched = False
        elif isinstance(audio, list):
            if len(audio) == 0:
                raise ValueError("Empty audio list provided")
            
            # Check if it's a list of file paths
            if all(isinstance(item, str) for item in audio):
                # Batch of audio file paths
                audio = [self._load_audio_from_path(path) for path in audio]
                is_batched = True
            else:
                # Check if it's batched audio arrays
                is_batched = isinstance(audio[0], (np.ndarray, list))
        else:
            # Single audio array or list
            is_batched = False
        
        # Process audio
        if is_batched:
            processed_audio = [self._process_single_audio(a) for a in audio]
        else:
            processed_audio = [self._process_single_audio(audio)]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            if len(processed_audio) == 1:
                # Create a proper batch dimension (B, T)
                input_features = torch.from_numpy(processed_audio[0]).unsqueeze(0).unsqueeze(1)
            else:
                # For batched input with different lengths, create a batch properly
                input_features = torch.stack([torch.from_numpy(a) for a in processed_audio]).unsqueeze(1)
        elif return_tensors == "np":
            if len(processed_audio) == 1:
                input_features = processed_audio[0][np.newaxis, np.newaxis, :]
            else:
                input_features = np.stack(processed_audio)[:, np.newaxis, :]
        else:
            input_features = processed_audio[0] if len(processed_audio) == 1 else processed_audio
        
        outputs = {
            "audio": input_features,  # Use "audio" instead of "input_features"
        }
        
        return outputs

    def _load_audio_from_path(self, audio_path: str) -> np.ndarray:
        """
        Load audio from file path.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.ndarray: Loaded audio array
        """
        # Get file extension to determine loading method
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        if file_ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            # Audio file - use librosa
            import librosa
            audio_array, sr = librosa.load(
                audio_path, 
                sr=self.sampling_rate, 
                mono=True
            )
            return audio_array
        elif file_ext == '.pt':
            # PyTorch tensor file
            audio_tensor = torch.load(audio_path, map_location='cpu', weights_only=True).squeeze()
            if isinstance(audio_tensor, torch.Tensor):
                audio_array = audio_tensor.numpy()
            else:
                audio_array = np.array(audio_tensor)
            return audio_array.astype(np.float32)
        elif file_ext == '.npy':
            # NumPy file
            audio_array = np.load(audio_path)
            return audio_array.astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: .wav, .mp3, .flac, .m4a, .ogg, .pt, .npy, .npz"
            )
    
    def preprocess_audio(
        self, 
        audio_path_or_array: Union[str, np.ndarray],
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Convenience method to preprocess audio from file path or array.
        This method is kept for backward compatibility but __call__ is recommended.
        
        Args:
            audio_path_or_array: Path to audio file or numpy array
            normalize: Whether to normalize (overrides default setting)
            
        Returns:
            np.ndarray: Preprocessed audio array
        """
        if isinstance(audio_path_or_array, str):
            audio_array = self._load_audio_from_path(audio_path_or_array)
        else:
            audio_array = np.array(audio_path_or_array, dtype=np.float32)
        
        # Override normalization setting if specified
        original_normalize = self.normalize_audio
        if normalize is not None:
            self.normalize_audio = normalize
        
        try:
            processed = self._process_single_audio(audio_array)
        finally:
            # Restore original setting
            self.normalize_audio = original_normalize
        
        return processed
    
    # Override to_dict method for configuration saving
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the object to a dict containing all attributes needed for serialization.
        """
        return self.feature_extractor_dict

    def save_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
        output_path: str = "output.wav",
        sampling_rate: Optional[int] = None,
        normalize: bool = False,
        batch_prefix: str = "audio_",
    ):
        """
        Save audio data to WAV file(s).
        
        Args:
            audio: Audio data to save. Can be:
                - torch.Tensor: PyTorch tensor with shape (B, C, T) or (B, T) or (T)
                - np.ndarray: NumPy array with shape (B, C, T) or (B, T) or (T)
                - List of tensors or arrays
            output_path: Path where to save the audio. If saving multiple files,
                this is treated as a directory and individual files will be saved inside.
            sampling_rate: Sampling rate for the saved audio. Defaults to the processor's rate.
            normalize: Whether to normalize audio before saving.
            batch_prefix: Prefix for batch files when saving multiple audios.
                
        Returns:
            List[str]: Paths to the saved audio files.
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required to save audio files. "
                "Install it with: pip install soundfile"
            )
        
        # Ensure audio is in the right format
        if isinstance(audio, torch.Tensor):
            # Convert PyTorch tensor to numpy
            audio_np = audio.float().detach().cpu().numpy()
        elif isinstance(audio, np.ndarray):
            audio_np = audio
        elif isinstance(audio, list):
            # Handle list of tensors or arrays
            if all(isinstance(a, torch.Tensor) for a in audio):
                audio_np = [a.float().detach().cpu().numpy() for a in audio]
            else:
                audio_np = audio
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        saved_paths = []
        
        # Handle based on shape or type
        if isinstance(audio_np, list):
            # Multiple separate audios to save
            output_dir = output_path
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each audio
            for i, audio_item in enumerate(audio_np):
                audio_item = self._prepare_audio_for_save(audio_item, normalize)
                file_path = os.path.join(output_dir, f"{batch_prefix}{i}.wav")
                sf.write(file_path, audio_item, sampling_rate)
                saved_paths.append(file_path)
                
        else:
            # Handle different dimensions
            if len(audio_np.shape) >= 3:  # (B, C, T) or similar
                # Get batch size
                batch_size = audio_np.shape[0]
                
                if batch_size > 1:
                    # Multiple audios in a batch
                    output_dir = output_path
                    
                    # Ensure output directory exists
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save each audio in the batch
                    for i in range(batch_size):
                        # Extract single audio and remove channel dim if present
                        single_audio = audio_np[i]
                        if len(single_audio.shape) > 1:
                            if single_audio.shape[0] == 1:  # (1, T)
                                single_audio = single_audio.squeeze(0)
                        
                        single_audio = self._prepare_audio_for_save(single_audio, normalize)
                        file_path = os.path.join(output_dir, f"{batch_prefix}{i}.wav")
                        sf.write(file_path, single_audio, sampling_rate)
                        saved_paths.append(file_path)
                else:
                    # Single audio with batch and channel dims
                    audio_item = audio_np.squeeze()  # Remove batch and channel dimensions
                    audio_item = self._prepare_audio_for_save(audio_item, normalize)
                    sf.write(output_path, audio_item, sampling_rate)
                    saved_paths.append(output_path)
            else:
                # Single audio without batch dimension
                audio_item = self._prepare_audio_for_save(audio_np, normalize)
                sf.write(output_path, audio_item, sampling_rate)
                saved_paths.append(output_path)
        
        return saved_paths

    def _prepare_audio_for_save(self, audio: np.ndarray, normalize: bool) -> np.ndarray:
        """
        Prepare audio for saving by ensuring it's the right shape and optionally normalizing.
        
        Args:
            audio: Audio data as numpy array
            normalize: Whether to normalize audio
            
        Returns:
            np.ndarray: Processed audio ready for saving
        """
        # Ensure right dimensionality
        if len(audio.shape) > 1 and audio.shape[0] == 1:  # (1, T)
            audio = audio.squeeze(0)
        
        # Normalize if requested
        if normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        
        return audio


AutoProcessor.register("VibeVoiceASRProcessor", VibeVoiceASRProcessor)

__all__ = [
    "VibeVoiceASRProcessor",
    "VibeVoiceTokenizerProcessor", 
    "AudioNormalizer",
    "load_audio_use_ffmpeg",
    "load_audio_bytes_use_ffmpeg",
    "COMMON_AUDIO_EXTS",
    "AUDIO_SAMPLE_RATE",
]
