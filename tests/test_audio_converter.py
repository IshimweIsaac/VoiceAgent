"""Tests for voice_agent.audio_converter — µ-law ↔ PCM + resampling."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from voice_agent.audio_converter import (
    ULAW_SAMPLE_RATE,
    PCM_SEND_RATE,
    PCM_RECEIVE_RATE,
    _pcm_to_ulaw,
    _ulaw_to_pcm,
    decode_twilio_audio,
    encode_twilio_audio,
    gemini_to_twilio,
    lin2ulaw,
    resample_pcm,
    twilio_to_gemini,
    ulaw2lin,
)

# ---------------------------------------------------------------------------
# Known µ-law reference values (computed from the G.711 algorithm)
# ---------------------------------------------------------------------------
# sample=0 → ulaw=255
_ZERO_ULAW = bytes([255, 255])
# sample=1 → ulaw
#   sign=0, sample=1+132=133, (133>>7)=1 → _EXP_LUT[1]=0, mant=(133>>3)&0xF=16&0xF=0
#   ulawbyte = ~0 = 255. Actually sample=1 is still quantized to 0 after bias+shift.
# Let's compute a non-trivial value via Python below.
# sample=8 → ulaw=254
_EIGHT_ULAW = bytes([254])


def _pcm16(*samples: int) -> bytes:
    """Helper: pack signed 16-bit integers into PCM s16le bytes."""
    return np.array(samples, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# lin2ulaw / ulaw2lin
# ---------------------------------------------------------------------------


class TestLin2UlAW:
    """Low-level µ-law encode tests."""

    def test_lin2ulaw_zero_produces_max_ulaw(self):
        """PCM sample 0 encodes to µ-law byte 255."""
        pcm = _pcm16(0)
        result = lin2ulaw(pcm)
        assert result == bytes([255])

    def test_lin2ulaw_known_sample_produces_expected(self):
        """A known PCM sample produces the expected µ-law bytes."""
        # sample=8 → ulaw=254 (computed from G.711 algorithm)
        pcm = _pcm16(8)
        result = lin2ulaw(pcm)
        assert result == _EIGHT_ULAW

    def test_lin2ulaw_negative_sample_produces_different_ulaw(self):
        """Negative samples have the sign bit set in µ-law."""
        pcm = _pcm16(-8)
        result = lin2ulaw(pcm)
        # -8 has sign bit set: ulawbyte = ~(0x80 | 0 | 1) & 0xFF = ~129 & 0xFF = 126
        assert result == bytes([126])

    def test_lin2ulaw_invalid_width_raises_error(self):
        """sample_width != 2 raises ValueError."""
        pcm = _pcm16(0, 100, -100)
        with pytest.raises(ValueError, match="Only 16-bit samples"):
            lin2ulaw(pcm, sample_width=1)

    def test_lin2ulaw_ulaw2lin_roundtrip(self):
        """PCM → µ-law → PCM preserves approximate value (lossy)."""
        raw = np.linspace(-5000, 5000, 50, dtype=np.int16)
        pcm_orig = raw.tobytes()
        ulaw = lin2ulaw(pcm_orig)
        pcm_back = ulaw2lin(ulaw)
        orig = np.frombuffer(pcm_orig, dtype=np.int16)
        back = np.frombuffer(pcm_back, dtype=np.int16)
        # µ-law is lossy; expect relative error < 10 %
        errors = np.abs(orig.astype(np.float32) - back.astype(np.float32))
        max_err = errors.max()
        assert max_err < 500, f"Max µ-law roundtrip error {max_err} >= 500"


class TestUlAW2Lin:
    """Low-level µ-law decode tests."""

    def test_ulaw2lin_zero_ulaw_produces_expected_pcm(self):
        """µ-law byte 0 decodes to a known PCM value.
        ulaw=0 → ~0 → 255, sign=0x80, exponent=0xF>>4=15... wait let me trace.

        Actually: ulawbyte=0 → ~0=255, sign=0x80, exponent=0xF=15... 15>7.
        Hmm let me check again: _ulaw_to_pcm(0):
          ulawbyte = (~0) & 0xFF = 255
          sign = 255 & 0x80 = 0x80 = 128
          exponent = (255 >> 4) & 0x07 = 15 & 0x07 = 7
          mantissa = 255 & 0x0F = 15
          sample = _DECODE_LUT[7] + (15 << (7 + 3)) = 16764 + (15 << 10) = 16764 + 15360 = 32124
          sign = 128, so sample = -32124
        """
        pcm = ulaw2lin(bytes([0]))
        assert len(pcm) == 2  # one 16-bit sample
        value = np.frombuffer(pcm, dtype=np.int16)[0]
        assert value == -32124

    def test_ulaw2lin_known_ulaw_produces_expected(self):
        """µ-law byte 254 decodes to PCM sample 8 (approximately)."""
        pcm = ulaw2lin(_EIGHT_ULAW)
        value = np.frombuffer(pcm, dtype=np.int16)[0]
        # Expected: 8
        assert value == 8

    def test_ulaw2lin_invalid_width_raises_error(self):
        """sample_width != 2 raises ValueError."""
        with pytest.raises(ValueError, match="Only 16-bit samples"):
            ulaw2lin(bytes([255, 200]), sample_width=1)

    def test_ulaw2lin_multiple_samples(self):
        """Decoding multiple µ-law bytes produces correct count."""
        ulaw = bytes([255, 254, 126])
        pcm = ulaw2lin(ulaw)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert len(samples) == 3


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


class TestResamplePCM:
    """PCM resampling tests."""

    def test_resample_pcm_same_rate_returns_original(self):
        """When from_rate == to_rate, returns data unchanged."""
        data = _pcm16(0, 100, -100, 200, -200)
        result = resample_pcm(data, 8000, 8000)
        assert result == data

    def test_resample_pcm_upsample_increases_length(self):
        """Resampling 8kHz→16kHz doubles data length."""
        # 0.1 seconds at 8kHz = 800 samples
        data = np.sin(np.linspace(0, 2 * np.pi * 10, 800)).astype(np.int16).tobytes()
        result = resample_pcm(data, 8000, 16000)
        orig_len = len(data) // 2
        new_len = len(result) // 2
        # 800 → 1600 (approximately, depends on rounding)
        assert new_len == 1600, f"Expected 1600 samples, got {new_len}"

    def test_resample_pcm_downsample_decreases_length(self):
        """Resampling 24kHz→8kHz reduces length."""
        # 0.1 seconds at 24kHz = 2400 samples
        data = np.sin(np.linspace(0, 2 * np.pi * 10, 2400)).astype(np.int16).tobytes()
        result = resample_pcm(data, 24000, 8000)
        orig_len = len(data) // 2
        new_len = len(result) // 2
        # 2400 → 800
        assert new_len == 800, f"Expected 800 samples, got {new_len}"

    def test_resample_pcm_invalid_rates_raises_error(self):
        """Zero or negative rates raise ValueError."""
        data = _pcm16(0, 100)
        with pytest.raises(ValueError, match="Sample rates must be positive"):
            resample_pcm(data, from_rate=0, to_rate=8000)
        with pytest.raises(ValueError, match="Sample rates must be positive"):
            resample_pcm(data, from_rate=8000, to_rate=-1)

    def test_resample_pcm_preserves_approximate_amplitude(self):
        """Resampling preserves signal amplitude roughly."""
        t = np.linspace(0, 0.05, 400, endpoint=False)
        data = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
        result = resample_pcm(data, 8000, 16000)
        orig = np.frombuffer(data, dtype=np.int16)
        res = np.frombuffer(result, dtype=np.int16)
        # Check neither is all zeros and max amplitude is in reasonable range
        assert np.max(np.abs(orig)) > 0
        assert np.max(np.abs(res)) > 1000


# ---------------------------------------------------------------------------
# High-level conversion functions
# ---------------------------------------------------------------------------


class TestTwilioToGemini:
    """Twilio µ-law → Gemini PCM conversion."""

    def test_twilio_to_gemini_returns_pcm_at_16khz(self):
        """Decoding + resampling produces correct output rate (inferred from length)."""
        # Create known PCM → µ-law → base64
        pcm_8khz = np.sin(np.linspace(0, 2 * np.pi * 10, 800)).astype(np.int16).tobytes()
        ulaw = lin2ulaw(pcm_8khz)
        payload_b64 = base64.b64encode(ulaw).decode("ascii")

        result = twilio_to_gemini(payload_b64)
        result_samples = len(result) // 2

        # 800 samples @ 8kHz → 1600 samples @ 16kHz
        assert result_samples == 1600, f"Expected 1600, got {result_samples}"

    def test_twilio_to_gemini_short_input(self):
        """Even a short input produces valid output."""
        pcm_8khz = _pcm16(100, 200, -100, -200)
        ulaw = lin2ulaw(pcm_8khz)
        payload_b64 = base64.b64encode(ulaw).decode("ascii")

        result = twilio_to_gemini(payload_b64)
        assert len(result) > 0
        assert len(result) % 2 == 0  # s16le, always even


class TestGeminiToTwilio:
    """Gemini PCM → Twilio µ-law conversion."""

    def test_gemini_to_twilio_returns_base64_ulaw(self):
        """Resampling + encoding produces a valid base64 string."""
        pcm_24khz = np.sin(np.linspace(0, 2 * np.pi * 10, 2400)).astype(np.int16).tobytes()
        result = gemini_to_twilio(pcm_24khz)

        assert isinstance(result, str)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_gemini_to_twilio_output_length_matches_input(self):
        """Verify the output µ-law length matches expected."""
        # 2400 samples @ 24kHz → 800 samples @ 8kHz → 800 µ-law bytes
        pcm_24khz = np.sin(np.linspace(0, 2 * np.pi * 10, 2400)).astype(np.int16).tobytes()
        result = gemini_to_twilio(pcm_24khz)
        decoded = base64.b64decode(result)
        # 800 samples → 800 µ-law bytes (1 byte per sample)
        assert len(decoded) == 800, f"Expected 800 µ-law bytes, got {len(decoded)}"


class TestRoundtrip:
    """Full roundtrip conversion tests."""

    def test_twilio_to_gemini_gemini_to_twilio_roundtrip(self):
        """Encode PCM to base64 µ-law → decode → compare approximate signal."""
        # Create original audio (0.1s @ 8kHz sine wave)
        t = np.linspace(0, 0.1, 800, endpoint=False)
        original = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
        pcm_8khz = original.tobytes()

        # Simulate Twilio→Server: encode to base64 µ-law
        ulaw = lin2ulaw(pcm_8khz)
        b64 = base64.b64encode(ulaw).decode("ascii")

        # Twilio → Gemini (8kHz µ-law → 16kHz PCM)
        gemini_pcm = twilio_to_gemini(b64)

        # Simulate Gemini response (24kHz PCM) — we have to create it
        # by resampling the original 8kHz → 24kHz to simulate Gemini output
        pcm_24khz_for_gemini = resample_pcm(pcm_8khz, 8000, 24000)

        # Gemini → Twilio (24kHz PCM → 8kHz base64 µ-law)
        back_b64 = gemini_to_twilio(pcm_24khz_for_gemini)

        # Decode back to PCM 8kHz
        back_pcm = ulaw2lin(base64.b64decode(back_b64))
        back_samples = np.frombuffer(back_pcm, dtype=np.int16)

        # Check shapes match
        assert len(back_samples) == len(original), (
            f"Length mismatch: {len(back_samples)} vs {len(original)}"
        )

        # µ-law is lossy, allow 15% relative error
        orig_f = original.astype(np.float32)
        back_f = back_samples.astype(np.float32)
        mask = orig_f != 0
        if mask.any():
            errors = np.abs((back_f[mask] - orig_f[mask]) / orig_f[mask])
            assert errors.mean() < 0.15, f"Mean relative error {errors.mean():.3f} >= 0.15"

    def test_encode_decode_twilio_audio_roundtrip(self):
        """Base64 encode PCM, then decode back — approximate signal preservation."""
        t = np.linspace(0, 0.1, 800, endpoint=False)
        original = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
        pcm_8khz = original.tobytes()

        # encode_twilio_audio: PCM → base64 µ-law
        b64 = encode_twilio_audio(pcm_8khz)

        # decode_twilio_audio: base64 µ-law → PCM
        back_pcm = decode_twilio_audio(b64)
        back = np.frombuffer(back_pcm, dtype=np.int16)

        assert len(back) == len(original)
        # Allow lossy compression error
        mae = np.mean(np.abs(back.astype(np.float32) - original.astype(np.float32)))
        assert mae < 1000, f"Mean absolute error {mae:.1f} >= 1000"


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_data_handled_gracefully(self):
        """Empty bytes produce empty or valid output."""
        assert lin2ulaw(b"") == b""
        assert ulaw2lin(b"") == b""
        assert resample_pcm(b"", 8000, 16000) == b""
        # Empty base64 decode produces empty ulaw → pcm
        assert decode_twilio_audio("") == b""
        # Empty PCM → encode should produce valid base64 of empty
        assert encode_twilio_audio(b"") == ""

    def test_decode_twilio_audio_empty_b64(self):
        """Empty base64 string decodes to empty PCM."""
        result = decode_twilio_audio("")
        assert result == b""

    def test_encode_twilio_audio_empty_pcm(self):
        """Empty PCM produces empty base64."""
        result = encode_twilio_audio(b"")
        assert result == ""

    def test_twilio_to_gemini_empty_b64(self):
        """Empty base64 yields empty PCM 16kHz."""
        result = twilio_to_gemini("")
        assert result == b""

    def test_gemini_to_twilio_empty_pcm(self):
        """Empty PCM 24kHz yields empty base64 string."""
        result = gemini_to_twilio(b"")
        # Empty PCM at 24kHz → resampled to 8kHz is still empty → ulaw empty → b64 empty
        assert result == ""

    def test_single_sample_lin2ulaw(self):
        """Single sample conversion works."""
        pcm = _pcm16(42)
        ulaw = lin2ulaw(pcm)
        assert len(ulaw) == 1
        back = ulaw2lin(ulaw)
        assert len(back) == 2

    def test_negative_pcm_roundtrip(self):
        """Negative PCM values survive the encode/decode cycle with correct sign for larger magnitudes."""
        pcm = _pcm16(-32123, -16000, -5000, 0, 5000, 16000, 32123)
        ulaw = lin2ulaw(pcm)
        back = ulaw2lin(ulaw)
        orig = np.frombuffer(pcm, dtype=np.int16)
        result = np.frombuffer(back, dtype=np.int16)
        assert len(result) == len(orig)
        # For larger magnitudes (>100), sign should be preserved
        mask = np.abs(orig) > 100
        assert np.all(
            np.sign(result[mask].astype(np.float32)) == np.sign(orig[mask].astype(np.float32))
        ), f"Sign mismatch on values: {list(zip(orig[mask], result[mask]))}"
