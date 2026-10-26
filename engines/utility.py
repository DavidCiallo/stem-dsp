"""工具类模块：Loudness Control / Normalize / Gain / Resample / Dither / Phase /
Azimuth / Signal Generator。全是确定性数学，不依赖模型。
"""

from fractions import Fraction

import numpy as np
import pyloudnorm as pyln
from scipy import signal as sps

from .base import Module, register

EPS = 1e-12


def _peak_db(x):
    return 20.0 * np.log10(max(float(np.max(np.abs(x))), EPS))


def _rms_db(x):
    return 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), EPS))


def _shift(x, n):
    """整数样本延迟，正数往后推。"""
    if n == 0:
        return x
    out = np.zeros_like(x)
    if n > 0:
        out[n:] = x[:len(x) - n]
    else:
        out[:n] = x[-n:]
    return out


@register
class Gain(Module):
    name = "Gain"
    params = [{"name": "gain_db", "type": "number", "default": 0.0, "min": -60.0, "max": 60.0}]

    def run(self, audio, sr, p):
        out = audio * 10.0 ** (p["gain_db"] / 20.0)
        return {"audio": out, "sr": sr,
                "metrics": {"gain_db": p["gain_db"], "peak_db": _peak_db(out)}}


@register
class Normalize(Module):
    name = "Normalize"
    params = [
        {"name": "mode", "type": "string", "default": "peak", "choices": ["peak", "rms"]},
        {"name": "target_db", "type": "number", "default": -1.0, "min": -60.0, "max": 0.0},
    ]

    def run(self, audio, sr, p):
        current = _peak_db(audio) if p["mode"] == "peak" else _rms_db(audio)
        gain = p["target_db"] - current
        out = audio * 10.0 ** (gain / 20.0)
        return {"audio": out, "sr": sr,
                "metrics": {"mode": p["mode"], "input_db": round(current, 2),
                            "gain_db": round(gain, 2), "peak_db": round(_peak_db(out), 2)}}


@register
class LoudnessControl(Module):
    name = "Loudness Control"
    params = [
        {"name": "target_lufs", "type": "number", "default": -14.0, "min": -40.0, "max": -5.0},
        {"name": "true_peak_db", "type": "number", "default": -1.0, "min": -6.0, "max": 0.0},
    ]

    def run(self, audio, sr, p):
        meter = pyln.Meter(sr)
        before = float(meter.integrated_loudness(audio))
        if not np.isfinite(before):
            raise ValueError("这段音频测不出响度（静音或太短）")
        out = audio * 10.0 ** ((p["target_lufs"] - before) / 20.0)

        # 真峰值：过采样 4 倍再看峰值，避免采样点之间的过冲被漏掉
        over = sps.resample_poly(out, 4, 1, axis=0)
        tp = 20.0 * np.log10(max(float(np.max(np.abs(over))), EPS))
        limited = False
        if tp > p["true_peak_db"]:
            out = out * 10.0 ** ((p["true_peak_db"] - tp) / 20.0)
            limited = True
            over = sps.resample_poly(out, 4, 1, axis=0)
            tp = 20.0 * np.log10(max(float(np.max(np.abs(over))), EPS))

        return {"audio": out, "sr": sr,
                "metrics": {"input_lufs": round(before, 2),
                            "output_lufs": round(float(meter.integrated_loudness(out)), 2),
                            "true_peak_db": round(tp, 2), "limited": limited}}


@register
class Resample(Module):
    name = "Resample"
    params = [{"name": "sample_rate", "type": "number", "default": 48000.0,
               "min": 8000.0, "max": 384000.0}]

    def run(self, audio, sr, p):
        target = int(round(p["sample_rate"]))
        if target == sr:
            return {"audio": audio, "sr": sr, "metrics": {"sample_rate": sr, "changed": False}}
        ratio = Fraction(target, sr).limit_denominator(1000)
        out = sps.resample_poly(audio, ratio.numerator, ratio.denominator, axis=0)
        return {"audio": out, "sr": target,
                "metrics": {"sample_rate": target, "changed": True,
                            "ratio": f"{ratio.numerator}/{ratio.denominator}"}}


@register
class Dither(Module):
    name = "Dither"
    params = [
        {"name": "bits", "type": "number", "default": 16.0, "min": 16.0, "max": 24.0},
        {"name": "shaping", "type": "boolean", "default": True},
    ]

    def run(self, audio, sr, p):
        bits = int(round(p["bits"]))
        lsb = 1.0 / (2 ** (bits - 1))
        rng = np.random.default_rng()

        # TPDF：两个均匀分布相减，峰值正好 1 LSB
        noise = (rng.random(audio.shape) - rng.random(audio.shape)) * lsb
        if p["shaping"]:
            # 抖动噪声本身做高通，把能量推向高频，人耳更不敏感
            noise = sps.lfilter([1.0, -1.0], [1.0], noise, axis=0) / np.sqrt(2.0)

        out = np.round((audio + noise) / lsb) * lsb
        return {"audio": out, "sr": sr, "subtype": f"PCM_{bits}",
                "metrics": {"bits": bits, "shaping": bool(p["shaping"]),
                            "peak_db": round(_peak_db(out), 2)}}


@register
class Phase(Module):
    name = "Phase"
    params = [
        {"name": "mode", "type": "string", "default": "rotate", "choices": ["rotate", "invert"]},
        {"name": "rotation_deg", "type": "number", "default": 90.0, "min": -180.0, "max": 180.0},
    ]

    def run(self, audio, sr, p):
        if p["mode"] == "invert":
            return {"audio": -audio, "sr": sr, "metrics": {"mode": "invert"}}
        theta = np.deg2rad(p["rotation_deg"])
        analytic = sps.hilbert(audio, axis=0)
        out = np.real(analytic * np.exp(1j * theta))
        return {"audio": out, "sr": sr,
                "metrics": {"mode": "rotate", "rotation_deg": p["rotation_deg"]}}


@register
class Azimuth(Module):
    name = "Azimuth"
    params = [
        {"name": "mode", "type": "string", "default": "auto", "choices": ["auto", "manual"]},
        {"name": "delay_ms", "type": "number", "default": 0.0, "min": -20.0, "max": 20.0},
        {"name": "max_delay_ms", "type": "number", "default": 10.0, "min": 0.1, "max": 50.0},
    ]

    def run(self, audio, sr, p):
        if audio.shape[1] < 2:
            raise ValueError("Azimuth 需要立体声，这段音频是单声道")

        if p["mode"] == "manual":
            n = int(round(p["delay_ms"] * sr / 1000.0))
            out = audio.copy()
            out[:, 1] = _shift(audio[:, 1], n)
            return {"audio": out, "sr": sr, "metrics": {"mode": "manual",
                                                        "delay_ms": p["delay_ms"]}}

        max_lag = int(round(p["max_delay_ms"] * sr / 1000.0))
        left, right = audio[:, 0], audio[:, 1]
        seg = slice(max_lag, len(left) - max_lag)
        lags = range(-max_lag, max_lag + 1)
        scores = [float(np.dot(left[seg], _shift(right, lag)[seg])) for lag in lags]
        lag = lags[int(np.argmax(scores))]

        out = audio.copy()
        if lag > 0:
            out[:, 1] = _shift(right, lag)
        elif lag < 0:
            out[:, 0] = _shift(left, -lag)
        return {"audio": out, "sr": sr,
                "metrics": {"mode": "auto", "delay_ms": round(lag * 1000.0 / sr, 3),
                            "delay_samples": lag}}


@register
class SignalGenerator(Module):
    name = "Signal Generator"
    needs_input = False
    params = [
        {"name": "waveform", "type": "string", "default": "sine",
         "choices": ["sine", "white", "pink", "sweep"]},
        {"name": "frequency", "type": "number", "default": 1000.0, "min": 20.0, "max": 20000.0},
        {"name": "sweep_to", "type": "number", "default": 20000.0, "min": 20.0, "max": 20000.0},
        {"name": "level_db", "type": "number", "default": -18.0, "min": -60.0, "max": 0.0},
        {"name": "duration", "type": "number", "default": 5.0, "min": 0.1, "max": 600.0},
        {"name": "sample_rate", "type": "number", "default": 48000.0,
         "min": 8000.0, "max": 384000.0},
    ]

    def run(self, audio, sr, p):
        sr = int(round(p["sample_rate"]))
        n = int(round(p["duration"] * sr))
        t = np.arange(n) / sr
        rng = np.random.default_rng()

        if p["waveform"] == "sine":
            x = np.sin(2 * np.pi * p["frequency"] * t)
        elif p["waveform"] == "white":
            x = rng.standard_normal(n)
        elif p["waveform"] == "pink":
            white = rng.standard_normal(n)
            spec = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n, 1.0 / sr)
            freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
            x = np.fft.irfft(spec / np.sqrt(freqs), n)
        else:
            f0, f1 = p["frequency"], p["sweep_to"]
            k = np.log(f1 / f0)
            phase = 2 * np.pi * f0 * p["duration"] / k * (np.exp(t / p["duration"] * k) - 1)
            x = np.sin(phase)

        x = x / max(float(np.max(np.abs(x))), EPS) * 10.0 ** (p["level_db"] / 20.0)
        out = x.reshape(-1, 1)
        return {"audio": out, "sr": sr,
                "metrics": {"waveform": p["waveform"], "sample_rate": sr,
                            "duration": round(n / sr, 3), "peak_db": round(_peak_db(out), 2)}}
