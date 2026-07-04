"""Audio conversion between Twilio µ-law 8kHz and Gemini PCM 16/24kHz.

Conversion chain:
  Twilio → Server:
    µ-law @ 8kHz (base64) → µ-law decode → PCM s16le @ 8kHz
    → linear resample to 16kHz → Gemini.send_realtime_input()

  Server → Twilio:
    Gemini response.data (PCM s16le @ 24kHz)
    → linear resample to 8kHz → µ-law encode → base64 → Twilio media

Since Python 3.13 removed the ``audioop`` module, µ-law encode/decode
are implemented directly using the standard G.711 algorithm from the
CPython audioop.c reference implementation.
"""

from __future__ import annotations

import base64

import numpy as np

ULAW_SAMPLE_RATE = 8000
PCM_SEND_RATE = 16000
PCM_RECEIVE_RATE = 24000

# ---------------------------------------------------------------------------
# µ-law (G.711) encode / decode
# Reference: CPython audioop.c (pre-3.13)
# ---------------------------------------------------------------------------

_BIAS = 0x84
_CLIP = 32635

# Exponent lookup table: index = (biased_sample >> 7) & 0xFF → exponent (0-7)
_EXP_LUT = [
    0,
    0,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
]

# Decode step table: base value for each exponent
_DECODE_LUT = [0, 132, 396, 924, 1980, 4092, 8316, 16764]


def _pcm_to_ulaw(sample: int) -> int:
    """Encode a single 16-bit PCM sample to an 8-bit µ-law byte.

    Args:
        sample: 16-bit signed PCM sample (-32768 to 32767).

    Returns:
        8-bit µ-law encoded byte (0-255).
    """
    sign = (sample >> 8) & 0x80
    if sign:
        sample = -sample
    if sample > _CLIP:
        sample = _CLIP

    sample += _BIAS
    exponent = _EXP_LUT[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulawbyte = ~(sign | (exponent << 4) | mantissa)
    return ulawbyte & 0xFF


def _ulaw_to_pcm(ulawbyte: int) -> int:
    """Decode a single 8-bit µ-law byte to a 16-bit PCM sample.

    Args:
        ulawbyte: 8-bit µ-law encoded byte (0-255).

    Returns:
        16-bit signed PCM sample.
    """
    ulawbyte = (~ulawbyte) & 0xFF
    sign = ulawbyte & 0x80
    exponent = (ulawbyte >> 4) & 0x07
    mantissa = ulawbyte & 0x0F
    sample = _DECODE_LUT[exponent] + (mantissa << (exponent + 3))
    if sign:
        sample = -sample
    return sample


def lin2ulaw(pcm_bytes: bytes, sample_width: int = 2) -> bytes:
    """Convert 16-bit linear PCM bytes to µ-law bytes.

    Args:
        pcm_bytes: Raw PCM audio data (s16le).
        sample_width: Byte width per sample (must be 2).

    Returns:
        µ-law encoded audio bytes.

    Raises:
        ValueError: If sample_width is not 2.
    """
    if sample_width != 2:
        raise ValueError("Only 16-bit samples (sample_width=2) are supported")

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    ulaw = np.array([_pcm_to_ulaw(int(s)) for s in samples], dtype=np.uint8)
    return ulaw.tobytes()


def ulaw2lin(ulaw_bytes: bytes, sample_width: int = 2) -> bytes:
    """Convert µ-law bytes to 16-bit linear PCM bytes.

    Args:
        ulaw_bytes: µ-law encoded audio bytes.
        sample_width: Byte width per output sample (must be 2).

    Returns:
        PCM s16le audio bytes.

    Raises:
        ValueError: If sample_width is not 2.
    """
    if sample_width != 2:
        raise ValueError("Only 16-bit samples (sample_width=2) are supported")

    ulaw = np.frombuffer(ulaw_bytes, dtype=np.uint8)
    pcm = np.array([_ulaw_to_pcm(int(b)) for b in ulaw], dtype=np.int16)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# High-level conversion functions
# ---------------------------------------------------------------------------


def decode_twilio_audio(payload_b64: str) -> bytes:
    """Decode Twilio's base64 µ-law to 16-bit PCM at 8kHz.

    Args:
        payload_b64: Base64-encoded µ-law audio bytes from Twilio.

    Returns:
        PCM s16le audio at 8kHz as raw bytes.
    """
    ulaw_bytes = base64.b64decode(payload_b64)
    return ulaw2lin(ulaw_bytes, 2)


def encode_twilio_audio(pcm_8khz: bytes) -> str:
    """Encode 8kHz PCM s16le to Twilio's base64 µ-law.

    Args:
        pcm_8khz: PCM s16le audio at 8kHz as raw bytes.

    Returns:
        Base64-encoded µ-law string ready for Twilio media payload.
    """
    ulaw_bytes = lin2ulaw(pcm_8khz, 2)
    return base64.b64encode(ulaw_bytes).decode("ascii")


def resample_pcm(data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM s16le audio from *from_rate* to *to_rate*.

    Uses numpy linear interpolation. For production, consider librosa
    or soxr for higher quality.

    Args:
        data: PCM s16le audio as raw bytes.
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.

    Returns:
        Resampled PCM s16le audio as raw bytes.

    Raises:
        ValueError: If *from_rate* or *to_rate* is not positive.
    """
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if from_rate == to_rate:
        return data

    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)

    indices = np.arange(new_length) / ratio
    left = indices.astype(np.int64)
    right = np.clip(left + 1, 0, len(samples) - 1)
    frac = (indices - left).astype(np.float32)

    resampled = samples[left] * (1.0 - frac) + samples[right] * frac
    return resampled.astype(np.int16).tobytes()


def twilio_to_gemini(payload_b64: str) -> bytes:
    """Convert a Twilio µ-law audio chunk to Gemini-ready PCM s16le 16kHz.

    Args:
        payload_b64: Base64-encoded µ-law chunk from Twilio.

    Returns:
        PCM s16le audio at 16kHz for Gemini Live API input.
    """
    pcm_8khz = decode_twilio_audio(payload_b64)
    return resample_pcm(pcm_8khz, ULAW_SAMPLE_RATE, PCM_SEND_RATE)


def gemini_to_twilio(pcm_24khz: bytes) -> str:
    """Convert Gemini PCM s16le 24kHz audio to Twilio base64 µ-law 8kHz.

    Args:
        pcm_24khz: PCM s16le audio at 24kHz from Gemini Live API.

    Returns:
        Base64-encoded µ-law string ready for Twilio media message.
    """
    pcm_8khz = resample_pcm(pcm_24khz, PCM_RECEIVE_RATE, ULAW_SAMPLE_RATE)
    return encode_twilio_audio(pcm_8khz)
