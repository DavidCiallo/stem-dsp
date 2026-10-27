"""RunPod serverless 分轨 worker。

收 sourceUrl（或 audioBase64），出六个分轨，zip 推到 uploadUrl。模型在模块
导入时加载一次，之后每次调用只做推理——冷启动那几秒整个 worker 生命周期只付一次。
"""

import base64
import os
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs.pretrained import get_model
from demucs.separate import load_track

MODEL_NAME = os.environ.get("STEM_MODEL", "htdemucs_6s")
STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = get_model(MODEL_NAME)
_model.eval()
_model.cpu()


def _download(url, dest):
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def _separate(src, work, wanted, shifts):
    """跟 demucs CLI 的 per-track 主体一致：同样的 load_track / apply_model / save_audio。"""
    wav = load_track(Path(src), _model.audio_channels, _model.samplerate)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(_model, wav[None], device=DEVICE, shifts=shifts,
                          split=True, overlap=0.25, progress=False,
                          num_workers=0, segment=None)[0]
    sources = sources * ref.std() + ref.mean()

    out = []
    for tensor, name in zip(sources, _model.sources):
        if wanted and name not in wanted:
            continue
        path = os.path.join(work, name + ".wav")
        save_audio(tensor, path, samplerate=_model.samplerate,
                   clip="rescale", bits_per_sample=16, as_float=False)
        out.append({"name": name, "bytes": os.path.getsize(path), "path": path})
    out.sort(key=lambda s: STEMS.index(s["name"]))
    return out


def _upload(path, url, method):
    raw = Path(path).read_bytes()
    req = urllib.request.Request(
        url, data=raw, method=(method or "PUT").upper(),
        headers={"Content-Type": "application/zip", "Content-Length": str(len(raw))})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.status


def handler(job):
    inp = job.get("input") or {}
    started = time.time()
    # 模型在模块导入时已经加载完，走到这里说明 worker 已经热了
    if inp.get("warmup"):
        return {"ok": True, "warmup": True, "model": MODEL_NAME, "device": DEVICE,
                "seconds": round(time.time() - started, 3)}
    wanted = [s for s in (inp.get("stems") or []) if s in STEMS]
    shifts = 2 if inp.get("quality") == "best" else 1

    with tempfile.TemporaryDirectory() as work:
        if inp.get("sourceUrl"):
            src = os.path.join(work, "source")
            _download(inp["sourceUrl"], src)
        elif inp.get("audioBase64"):
            src = os.path.join(work, "source")
            with open(src, "wb") as f:
                f.write(base64.b64decode(inp["audioBase64"]))
        else:
            raise ValueError("需要 sourceUrl 或 audioBase64")

        stems = _separate(src, work, wanted, shifts)
        if not stems:
            raise ValueError(f"没有产出分轨，检查 stems 取值 {list(STEMS)}")

        zip_path = os.path.join(work, "all.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for s in stems:
                z.write(s["path"], s["name"] + ".wav")
        zip_bytes = os.path.getsize(zip_path)

        out = {
            "ok": True,
            "taskId": inp.get("taskId") or job.get("id"),
            "duration": round(sf.info(stems[0]["path"]).duration, 3),
            "seconds": round(time.time() - started, 2),
            "stems": [{"name": s["name"], "bytes": s["bytes"]} for s in stems],
            "zip": {"bytes": zip_bytes},
        }
        if not inp.get("uploadUrl"):
            raise ValueError("需要 uploadUrl，分轨结果只能推到对象存储")
        out["uploadStatus"] = _upload(zip_path, inp["uploadUrl"], inp.get("uploadMethod"))
        out["uploadUrl"] = inp["uploadUrl"]
    return out


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
