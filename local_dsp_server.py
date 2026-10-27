"""本地音频 HTTP 服务：把 StemDeck 的分轨能力包成接口，给 StreamReady 调用。

    python local_dsp_server.py            # 默认 127.0.0.1:8765

设了 DSP_DEMUCS_PYTHON 就直接调 demucs，只出分轨；否则转发给 StemDeck
（StemDeck.exe，监听 https://127.0.0.1:8443），分轨在那边完成。
"""

import base64
import json
import math
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf

import engines

HOST = os.environ.get("DSP_HOST", "127.0.0.1")
PORT = int(os.environ.get("DSP_PORT", "8765"))
STEMDECK = os.environ.get("STEMDECK_URL", "https://127.0.0.1:8443").rstrip("/")
TOKEN = os.environ.get("DSP_TOKEN", "")
UPLOAD_DIR = os.environ.get("DSP_UPLOAD_DIR") or os.path.join(tempfile.gettempdir(), "dsp-uploads")

# 指向装了 torch+demucs 的解释器就绕开 StemDeck 整条 pipeline，只跑分离
DEMUCS_PYTHON = os.environ.get("DSP_DEMUCS_PYTHON", "")
DEMUCS_MODEL = os.environ.get("DSP_DEMUCS_MODEL", "htdemucs_6s")
DEMUCS_DEVICE = os.environ.get("DSP_DEMUCS_DEVICE", "cuda")

# 配齐 bucket/endpoint/ak/sk 就把结果直接推进 OSS，不再留本地也不依赖 /result
OSS_BUCKET = os.environ.get("DSP_OSS_BUCKET", "")
OSS_ENDPOINT = os.environ.get("DSP_OSS_ENDPOINT", "")
OSS_AK = os.environ.get("DSP_OSS_AK", "")
OSS_SK = os.environ.get("DSP_OSS_SK", "")
OSS_PREFIX = os.environ.get("DSP_OSS_PREFIX", "stems")
OSS_EXPIRES = int(os.environ.get("DSP_OSS_EXPIRES", str(7 * 24 * 3600)))

# StemDeck 用自签证书，本机通信不走证书链校验
_SSL = ssl._create_unverified_context()

STEM_NAMES = ("vocals", "drums", "bass", "guitar", "piano", "other")
# Content-Type 的子类型不是文件后缀，audio/mpeg 存成 .mpeg 会被 StemDeck 拒收
_UPLOAD_EXT = {"mpeg": "mp3", "x-wav": "wav", "wave": "wav", "x-flac": "flac", "x-m4a": "m4a"}
POLL_SECONDS = 3
JOB_TIMEOUT = 1800

_jobs = {}
_counter = [0]
_lock = threading.Lock()


def _next_id():
    _counter[0] += 1
    return f"dsp-{_counter[0]}"


def _urlopen(req, timeout):
    return urllib.request.urlopen(req, timeout=timeout, context=_SSL)


def sd_health():
    with _urlopen(STEMDECK + "/api/health", 15) as r:
        return json.loads(r.read())


def _multipart(filename, data):
    boundary = "dsp" + uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    return head + data + f"\r\n--{boundary}--\r\n".encode("utf-8"), \
        f"multipart/form-data; boundary={boundary}"


def sd_submit(path):
    with open(path, "rb") as f:
        data = f.read()
    body, ctype = _multipart(os.path.basename(path), data)
    req = urllib.request.Request(STEMDECK + "/api/jobs", data=body, method="POST",
                                 headers={"Content-Type": ctype})
    with _urlopen(req, 900) as r:
        return json.loads(r.read())["job_id"]


def sd_job(job_id):
    with _urlopen(f"{STEMDECK}/api/jobs/{job_id}", 60) as r:
        return json.loads(r.read())


def sd_download(path, dest):
    with _urlopen(STEMDECK + path, 900) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _save_remote(url, dest):
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _bytes(path):
    return os.path.getsize(path) if os.path.isfile(path) else 0


DOCS = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>本地音频服务 API</title>
<style>
  :root { --fg:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; --accent:#2563eb; --code:#f3f4f6; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif; }
  .wrap { max-width:960px; margin:0 auto; padding:48px 24px 96px; }
  header { border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:32px; }
  h1 { font-size:28px; font-weight:650; margin:0 0 8px; letter-spacing:-.01em; }
  .sub { color:var(--muted); font-size:14px; }
  h2 { font-size:20px; font-weight:600; margin:56px 0 8px; padding-top:8px; }
  h3 { font-size:14px; font-weight:600; margin:24px 0 8px; color:var(--muted);
       text-transform:uppercase; letter-spacing:.06em; }
  p { margin:8px 0; }
  code { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
         background:var(--code); padding:2px 6px; border-radius:4px; font-size:13px; }
  table { width:100%; border-collapse:collapse; margin:12px 0 4px; font-size:14px; }
  th { text-align:left; font-weight:600; color:var(--muted); font-size:12px;
       text-transform:uppercase; letter-spacing:.06em; padding:8px 12px;
       border-bottom:1px solid var(--line); }
  td { padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
  td:first-child { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
                   font-size:13px; white-space:nowrap; }
  td.req { color:var(--muted); white-space:nowrap; }
  .endpoint { display:flex; align-items:center; gap:12px; margin:56px 0 4px;
              padding-top:8px; border-top:1px solid var(--line); }
  .endpoint h2 { margin:0; padding:0; border:0; font-size:18px;
                 font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  .m { font-size:11px; font-weight:700; letter-spacing:.06em; padding:3px 8px;
       border-radius:4px; color:#fff; }
  .get { background:#059669; } .post { background:var(--accent); }
  .note { color:var(--muted); font-size:13px; }
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>本地音频服务 API</h1>
  <div class="sub">JSON over HTTP · 路径相对于服务根地址 · 单文件整段处理</div>
</header>

<h2>鉴权</h2>
<p class="note">除 <code>GET /health</code> 与 <code>GET /api/docs</code> 外，所有请求必须携带以下任一头字段：</p>
<table>
<thead><tr><th>Header</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Authorization</td><td><code>Bearer &lt;DSP_TOKEN&gt;</code></td></tr>
<tr><td>X-DSP-Token</td><td><code>&lt;DSP_TOKEN&gt;</code></td></tr>
</tbody>
</table>

<h2>端点</h2>
<table>
<thead><tr><th>Method</th><th>Path</th><th>说明</th></tr></thead>
<tbody>
<tr><td>GET</td><td>/health</td><td>服务状态</td></tr>
<tr><td>GET</td><td>/api/docs</td><td>本文档</td></tr>
<tr><td>GET</td><td>/capabilities</td><td>模块与操作清单</td></tr>
<tr><td>POST</td><td>/upload</td><td>上传源音频存档</td></tr>
<tr><td>POST</td><td>/process</td><td>提交处理</td></tr>
<tr><td>POST</td><td>/warmup</td><td>预热：跑一次 1 秒的分离</td></tr>
<tr><td>GET</td><td>/task/{taskId}</td><td>任务状态</td></tr>
<tr><td>GET</td><td>/tasks</td><td>最近任务</td></tr>
<tr><td>GET</td><td>/result/{taskId}/out.wav</td><td>本地模块输出</td></tr>
<tr><td>GET</td><td>/result/{taskId}/all.zip</td><td>分轨打包</td></tr>
<tr><td>GET</td><td>/result/{taskId}/{stem}.wav</td><td>单个音轨</td></tr>
</tbody>
</table>

<div class="endpoint"><span class="m get">GET</span><h2>/health</h2></div>
<p class="note">无需鉴权。</p>
<h3>响应</h3>
<table>
<thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
<tbody>
<tr><td>ok</td><td>boolean</td><td>服务可用</td></tr>
<tr><td>engine</td><td>string</td><td>分离引擎标识：<code>demucs</code> 或 <code>stemdeck</code></td></tr>
<tr><td>engine_ok</td><td>boolean</td><td>分离引擎可用状态</td></tr>
<tr><td>version</td><td>string</td><td>StemDeck 版本，仅 <code>stemdeck</code> 引擎返回</td></tr>
<tr><td>model</td><td>string</td><td>分轨模型</td></tr>
<tr><td>device</td><td>string</td><td>推理设备</td></tr>
<tr><td>tasks</td><td>number</td><td>已登记任务数</td></tr>
<tr><td>engine_error</td><td>string</td><td>分离引擎不可达时的原因</td></tr>
</tbody>
</table>

<div class="endpoint"><span class="m post">POST</span><h2>/warmup</h2></div>
<p class="note">需要鉴权。跑一次 1 秒音频的完整分离，把 torch、CUDA、模型加载这些一次性初始化做完，之后的任务只剩推理。容器首次启动要跑 50 秒左右，模型已加载时几秒返回。</p>
<h3>响应</h3>
<table>
<thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
<tbody>
<tr><td>ok</td><td>boolean</td><td>预热成功</td></tr>
<tr><td>model</td><td>string</td><td>分轨模型</td></tr>
<tr><td>device</td><td>string</td><td>推理设备</td></tr>
<tr><td>seconds</td><td>number</td><td>本次预热耗时</td></tr>
</tbody>
</table>

<div class="endpoint"><span class="m get">GET</span><h2>/capabilities</h2></div>
<p class="note">需要鉴权。</p>
<h3>响应</h3>
<table>
<thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
<tbody>
<tr><td>ok</td><td>boolean</td><td>请求成功</td></tr>
<tr><td>modules</td><td>array</td><td>本地引擎模块</td></tr>
<tr><td>modules[].module</td><td>string</td><td>模块名，作为 <code>module</code> 传入</td></tr>
<tr><td>modules[].kind</td><td>string</td><td>处理粒度，固定 <code>whole_file</code></td></tr>
<tr><td>modules[].needsInput</td><td>boolean</td><td>是否需要源音频</td></tr>
<tr><td>modules[].parameters</td><td>array</td><td>参数定义</td></tr>
<tr><td>parameters[].name</td><td>string</td><td>参数名</td></tr>
<tr><td>parameters[].type</td><td>string</td><td><code>number</code> / <code>boolean</code> / <code>string</code></td></tr>
<tr><td>parameters[].default</td><td>any</td><td>缺省值</td></tr>
<tr><td>parameters[].min</td><td>number</td><td>数值下限，超出按边界处理</td></tr>
<tr><td>parameters[].max</td><td>number</td><td>数值上限，超出按边界处理</td></tr>
<tr><td>parameters[].choices</td><td>array</td><td>枚举取值，超出报 400</td></tr>
<tr><td>operations</td><td>array</td><td>分离操作</td></tr>
<tr><td>operations[].name</td><td>string</td><td>固定 <code>separate</code></td></tr>
<tr><td>operations[].needsInput</td><td>boolean</td><td>固定 true</td></tr>
<tr><td>operations[].stems</td><td>array</td><td>可用音轨名</td></tr>
</tbody>
</table>

<div class="endpoint"><span class="m post">POST</span><h2>/upload</h2></div>
<p class="note">需要鉴权。请求体为音频原始字节，<code>Content-Type</code> 决定存档扩展名。返回的 <code>fileId</code> 可在 <code>/process</code> 中反复使用，之后的调用不再传输音频。</p>
<h3>响应</h3>
<table>
<thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
<tbody>
<tr><td>ok</td><td>boolean</td><td>上传成功</td></tr>
<tr><td>fileId</td><td>string</td><td>存档标识，32 位十六进制</td></tr>
<tr><td>bytes</td><td>number</td><td>已接收字节数</td></tr>
<tr><td>format</td><td>string</td><td>存档扩展名</td></tr>
<tr><td>duration</td><td>number</td><td>音频时长，秒</td></tr>
</tbody>
</table>

<div class="endpoint"><span class="m post">POST</span><h2>/process</h2></div>
<p class="note">需要鉴权。同步处理，完成或失败后返回。带 <code>module</code> 执行本地模块，不带 <code>module</code> 执行分离。</p>
<h3>请求字段</h3>
<table>
<thead><tr><th>字段</th><th>类型</th><th>必填</th><th>说明</th></tr></thead>
<tbody>
<tr><td>module</td><td>string</td><td class="req">否</td><td>本地引擎模块名，取值见 <code>/capabilities</code></td></tr>
<tr><td>params</td><td>object</td><td class="req">否</td><td>模块参数，缺省项取 default</td></tr>
<tr><td>fileId</td><td>string</td><td class="req">否</td><td>已存档音频的标识，由 <code>/upload</code> 返回</td></tr>
<tr><td>sourceUrl</td><td>string</td><td class="req">否</td><td>源音频 HTTP(S) 地址，由服务端下载</td></tr>
<tr><td>path</td><td>string</td><td class="req">否</td><td>服务端本机文件路径</td></tr>
<tr><td>audioBase64</td><td>string</td><td class="req">否</td><td>源音频 Base64</td></tr>
<tr><td>format</td><td>string</td><td class="req">否</td><td><code>audioBase64</code> 解码格式，默认 <code>wav</code></td></tr>
<tr><td>stems</td><td>array</td><td class="req">否</td><td>分离音轨，缺省为全部</td></tr>
<tr><td>uploadUrl</td><td>string</td><td class="req">否</td><td>结果上传地址</td></tr>
<tr><td>uploadMethod</td><td>string</td><td class="req">否</td><td>上传方法，默认 <code>PUT</code></td></tr>
<tr><td>taskId</td><td>string</td><td class="req">否</td><td>任务标识，缺省由服务端生成</td></tr>
<tr><td>async</td><td>boolean</td><td class="req">否</td><td>true 时立即返回任务号，默认 false</td></tr>
</tbody>
</table>
<p class="note"><code>fileId</code>、<code>sourceUrl</code>、<code>path</code>、<code>audioBase64</code> 四选一。模块的 <code>needsInput</code> 为 true 时必须提供其一。</p>

<h3>响应 · 本地模块</h3>
<table>
<thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
<tbody>
<tr><td>ok</td><td>boolean</td><td>处理成功</td></tr>
<tr><td>taskId</td><td>string</td><td>任务标识</td></tr>
<tr><td>status</td><td>string</td><td>固定 <code>done</code></td></tr>
<tr><td>module</td><td>string</td><td>模块名</td></tr>
<tr><td>params</td><td>object</td><td>实际生效的参数</td></tr>
<tr><td>sampleRate</td><td>number</td><td>输出采样率</td></tr>
<tr><td>channels</td><td>number</td><td>输出声道数</td></tr>
<tr><td>duration</td><td>number</td><td>输出时长，秒</td></tr>
<tr><td>seconds</td><td>number</td><td>处理耗时，秒</td></tr>
<tr><td>metrics</td><td>object</td><td>模块指标，字段随模块不同</td></tr>
<tr><td>output.bytes</td><td>number</td><td>输出文件字节数</td></tr>
<tr><td>output.url</td><td>string</td><td>输出下载路径</td></tr>
<tr><td>deduplicated</td><td>boolean</td><td>是否为重复 taskId 的历史结果</td></tr>
<tr><td>uploadUrl</td><td>string</td><td>上传地址，仅在传入 uploadUrl 时出现</td></tr>
<tr><td>uploadStatus</td><td>number</td><td>上传返回的 HTTP 状态码</td></tr>
</tbody>
</table>

<h3>响应 · 分离</h3>
<table>
<thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
<tbody>
<tr><td>ok</td><td>boolean</td><td>处理成功</td></tr>
<tr><td>taskId</td><td>string</td><td>任务标识</td></tr>
<tr><td>status</td><td>string</td><td>固定 <code>done</code></td></tr>
<tr><td>operation</td><td>string</td><td>固定 <code>separate</code></td></tr>
<tr><td>jobId</td><td>string</td><td>分离引擎任务号，仅 <code>stemdeck</code> 引擎返回</td></tr>
<tr><td>duration</td><td>number</td><td>源音频时长，秒</td></tr>
<tr><td>seconds</td><td>number</td><td>处理耗时，秒</td></tr>
<tr><td>stems</td><td>array</td><td>音轨列表</td></tr>
<tr><td>stems[].name</td><td>string</td><td>音轨名</td></tr>
<tr><td>stems[].bytes</td><td>number</td><td>文件字节数</td></tr>
<tr><td>stems[].url</td><td>string</td><td>下载地址，配置了 OSS 时为预签名 URL</td></tr>
<tr><td>zip.bytes</td><td>number</td><td>打包文件字节数</td></tr>
<tr><td>zip.url</td><td>string</td><td>打包下载路径</td></tr>
<tr><td>deduplicated</td><td>boolean</td><td>是否为重复 taskId 的历史结果</td></tr>
<tr><td>uploadUrl</td><td>string</td><td>上传地址，仅在传入 uploadUrl 时出现</td></tr>
<tr><td>uploadStatus</td><td>number</td><td>上传返回的 HTTP 状态码</td></tr>
</tbody>
</table>

<div class="endpoint"><span class="m get">GET</span><h2>/task/{taskId}</h2></div>
<p class="note">需要鉴权。</p>
<h3>响应</h3>
<table>
<thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
<tbody>
<tr><td>ok</td><td>boolean</td><td>请求成功</td></tr>
<tr><td>taskId</td><td>string</td><td>任务标识</td></tr>
<tr><td>status</td><td>string</td><td><code>queued</code> / <code>running</code> / <code>done</code> / <code>failed</code></td></tr>
<tr><td>stage</td><td>string</td><td>当前阶段</td></tr>
<tr><td>progress</td><td>number</td><td>进度，0 到 1</td></tr>
<tr><td>jobId</td><td>string</td><td>分离引擎任务号，仅 <code>stemdeck</code> 引擎的分离任务提供</td></tr>
<tr><td>created</td><td>number</td><td>创建时间，Unix 秒</td></tr>
<tr><td>elapsed</td><td>number</td><td>总耗时，秒</td></tr>
<tr><td>result</td><td>object</td><td>完成后与 <code>/process</code> 响应相同</td></tr>
<tr><td>error</td><td>string</td><td>失败原因</td></tr>
</tbody>
</table>

<div class="endpoint"><span class="m get">GET</span><h2>/tasks</h2></div>
<p class="note">需要鉴权。</p>
<h3>响应</h3>
<table>
<thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
<tbody>
<tr><td>ok</td><td>boolean</td><td>请求成功</td></tr>
<tr><td>tasks</td><td>array</td><td>最近 50 个任务，字段同 <code>/task/{taskId}</code></td></tr>
</tbody>
</table>

<div class="endpoint"><span class="m get">GET</span><h2>/result/{taskId}/out.wav</h2></div>
<p class="note">需要鉴权。本地模块输出音频，<code>Content-Type: audio/wav</code>。</p>

<div class="endpoint"><span class="m get">GET</span><h2>/result/{taskId}/all.zip</h2></div>
<p class="note">需要鉴权。分离结果打包，<code>Content-Type: application/zip</code>。</p>

<div class="endpoint"><span class="m get">GET</span><h2>/result/{taskId}/{stem}.wav</h2></div>
<p class="note">需要鉴权。单个音轨，<code>Content-Type: audio/wav</code>。<code>stem</code> 取值见 <code>/capabilities</code> 的 <code>operations[].stems</code>。</p>

<h2>状态码</h2>
<table>
<thead><tr><th>Code</th><th>说明</th></tr></thead>
<tbody>
<tr><td>200</td><td>成功</td></tr>
<tr><td>400</td><td>参数错误，响应体含 <code>error</code></td></tr>
<tr><td>401</td><td>缺少或错误的 token</td></tr>
<tr><td>404</td><td>任务或结果不存在，以及结果尚未生成</td></tr>
</tbody>
</table>

<h2>处理范围</h2>
<table>
<thead><tr><th>项</th><th>说明</th></tr></thead>
<tbody>
<tr><td>输出格式</td><td>WAV，24-bit；<code>Dither</code> 指定 16 bit 时输出 16-bit</td></tr>
<tr><td>采样率</td><td><code>Resample</code> 与 <code>Signal Generator</code> 决定输出采样率，其余模块沿用源文件采样率</td></tr>
<tr><td>声道</td><td>沿用源文件声道数；<code>Signal Generator</code> 固定单声道</td></tr>
<tr><td>分离限制</td><td><code>stemdeck</code> 引擎受 StemDeck 的上传 400 MB、时长 1200 秒约束；<code>demucs</code> 引擎无此限制</td></tr>
</tbody>
</table>

</div>
</body>
</html>
"""


def capabilities():
    return {
        "ok": True,
        "modules": engines.describe_all(),
        "operations": [{"name": "separate", "kind": "whole_file", "needsInput": True,
                        "stems": list(STEM_NAMES)}],
    }


def _resolve_in(body):
    if body.get("fileId"):
        file_id = str(body["fileId"])
        if not re.fullmatch(r"[0-9a-f]{32}", file_id):
            raise ValueError(f"fileId 无效：{file_id}")
        hits = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(file_id + ".")] if os.path.isdir(UPLOAD_DIR) else []
        if not hits:
            raise ValueError(f"找不到存档文件：{file_id}")
        return os.path.join(UPLOAD_DIR, hits[0])
    if body.get("sourceUrl"):
        url = body["sourceUrl"]
        suffix = os.path.splitext(url.split("?")[0])[1] or "." + (body.get("format") or "wav")
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        _save_remote(url, path)
        return path
    if body.get("path"):
        path = body["path"]
        if not os.path.isfile(path):
            raise ValueError(f"找不到文件：{path}")
        return path
    if body.get("audioBase64"):
        fd, path = tempfile.mkstemp(suffix="." + (body.get("format") or "wav"))
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(body["audioBase64"]))
        return path
    raise ValueError("需要 fileId、sourceUrl、path 或 audioBase64 之一")


def _upload(path, url, method, ctype):
    with open(path, "rb") as f:
        raw = f.read()
    req = urllib.request.Request(url, data=raw, method=(method or "PUT").upper(),
                                 headers={"Content-Type": ctype,
                                          "Content-Length": str(len(raw))})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status


_oss_bucket_cache = [None]


def _oss_bucket():
    if _oss_bucket_cache[0] is None:
        import oss2

        if not (OSS_BUCKET and OSS_ENDPOINT and OSS_AK and OSS_SK):
            raise RuntimeError("OSS 配置不全，需要 DSP_OSS_BUCKET / DSP_OSS_ENDPOINT / DSP_OSS_AK / DSP_OSS_SK")
        endpoint = OSS_ENDPOINT if "://" in OSS_ENDPOINT else "https://" + OSS_ENDPOINT
        _oss_bucket_cache[0] = oss2.Bucket(oss2.Auth(OSS_AK, OSS_SK), endpoint, OSS_BUCKET)
    return _oss_bucket_cache[0]


def _publish_oss(task_id, workdir, result):
    """分轨和 zip 推进 OSS，把 url 换成预签名地址。容器里因此不需要保留结果文件。"""
    bucket = _oss_bucket()
    prefix = f"{OSS_PREFIX}/{task_id}"
    for stem in result.get("stems") or []:
        key = f"{prefix}/{stem['name']}.wav"
        bucket.put_object_from_file(key, os.path.join(workdir, stem["name"] + ".wav"),
                                    headers={"Content-Type": "audio/wav"})
        stem["url"] = bucket.sign_url("GET", key, OSS_EXPIRES)
    zip_path = os.path.join(workdir, "all.zip")
    if os.path.isfile(zip_path):
        key = f"{prefix}/all.zip"
        bucket.put_object_from_file(key, zip_path, headers={"Content-Type": "application/zip"})
        result["zip"]["url"] = bucket.sign_url("GET", key, OSS_EXPIRES)
    return {"bucket": OSS_BUCKET, "prefix": prefix}


def _stem_key(filename):
    # StemDeck 打包时给的是 vocals.wav，有时带源文件名前缀，取下划线后那段
    return os.path.splitext(filename)[0].rsplit("_", 1)[-1]


def _collect(task_id, job_id, wanted, workdir):
    """把分轨结果落到本地，返回对外可见的音轨清单。"""
    zip_path = os.path.join(workdir, "all.zip")
    sd_download(f"/api/jobs/{job_id}/stems/all.zip?format=wav", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(workdir)

    found = []
    for path in os.listdir(workdir):
        if not path.lower().endswith(".wav"):
            continue
        name = _stem_key(path)
        if wanted and name not in wanted:
            continue
        found.append({"name": name,
                      "bytes": _bytes(os.path.join(workdir, path)),
                      "url": f"/result/{task_id}/{name}.wav"})
    found.sort(key=lambda s: STEM_NAMES.index(s["name"]) if s["name"] in STEM_NAMES else 99)
    return found, zip_path


def _progress(task_id, **fields):
    with _lock:
        _jobs[task_id].update(fields)


def _run_local(task_id, body, workdir, cls):
    _progress(task_id, stage="Processing...")
    params = cls.resolve(body.get("params") or {})
    if cls.needs_input:
        src = _resolve_in(body)
        audio, sr = sf.read(src, always_2d=True, dtype="float64")
    else:
        audio, sr = None, 0

    done = cls().run(audio, sr, params)
    path = os.path.join(workdir, "out.wav")
    sf.write(path, done["audio"], done["sr"], subtype=done.get("subtype", "PCM_24"))

    frames, channels = done["audio"].shape
    return {
        "ok": True,
        "taskId": task_id,
        "status": "done",
        "module": cls.name,
        "params": params,
        "sampleRate": done["sr"],
        "channels": int(channels),
        "duration": round(frames / done["sr"], 3),
        "metrics": done.get("metrics", {}),
        "output": {"bytes": _bytes(path), "url": f"/result/{task_id}/out.wav"},
        "__upload": (path, "audio/wav"),
    }


def _demucs_separate(task_id, src, workdir, wanted):
    """直接跑 demucs。StemDeck 那条 pipeline 里 analyze / post / beatgrid 占掉
    大半时间，出的却是 BPM、调性、节拍网格这类元数据，这里一概不要。"""
    _progress(task_id, status="running", stage="Separating stems...")
    out = os.path.join(workdir, "demucs")
    cmd = [DEMUCS_PYTHON, "-m", "demucs", "-n", DEMUCS_MODEL,
           "-d", DEMUCS_DEVICE, "-o", out, src]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"demucs 退出码 {proc.returncode}：{proc.stderr.strip()[-400:]}")

    stems = []
    for root, _, files in os.walk(out):
        for name in files:
            stem = _stem_key(name)
            if not name.lower().endswith(".wav") or stem not in STEM_NAMES:
                continue
            if wanted and stem not in wanted:
                continue
            dest = os.path.join(workdir, stem + ".wav")
            shutil.move(os.path.join(root, name), dest)
            stems.append({"name": stem, "bytes": _bytes(dest),
                          "url": f"/result/{task_id}/{stem}.wav"})
    if not stems:
        raise RuntimeError(f"demucs 没有产出分轨，检查模型名 {DEMUCS_MODEL}")
    stems.sort(key=lambda s: STEM_NAMES.index(s["name"]))

    zip_path = os.path.join(workdir, "all.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for s in stems:
            z.write(os.path.join(workdir, s["name"] + ".wav"), s["name"] + ".wav")
    return {
        "ok": True,
        "taskId": task_id,
        "status": "done",
        "operation": "separate",
        "duration": round(sf.info(os.path.join(workdir, stems[0]["name"] + ".wav")).duration, 3),
        "stems": stems,
        "zip": {"bytes": _bytes(zip_path), "url": f"/result/{task_id}/all.zip"},
        "__upload": (zip_path, "application/zip"),
    }


def _warmup():
    """跑一次 1 秒的分离，把 torch、CUDA、模型、显存全部热起来。

    冷启动那 50 秒基本都花在这些初始化上，跑完这一次，后面每单就只剩推理。
    """
    if not DEMUCS_PYTHON:
        raise RuntimeError("warmup 只对 demucs 引擎有效")
    started = time.time()
    work = tempfile.mkdtemp(prefix="dsp-warmup-")
    try:
        src = os.path.join(work, "warmup.wav")
        # 纯静音会让 demucs 的归一化除以零，用极低幅度的正弦代替
        tone = [[0.01 * math.sin(2 * math.pi * 440 * i / 44100)] * 2 for i in range(44100)]
        sf.write(src, tone, 44100, subtype="PCM_16")
        cmd = [DEMUCS_PYTHON, "-m", "demucs", "-n", DEMUCS_MODEL, "-d", DEMUCS_DEVICE,
               "-o", os.path.join(work, "out"), src]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"demucs 退出码 {proc.returncode}：{proc.stderr.strip()[-400:]}")
        return {"ok": True, "model": DEMUCS_MODEL, "device": DEMUCS_DEVICE,
                "seconds": round(time.time() - started, 2)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_separate(task_id, body, workdir):
    _progress(task_id, status="running", stage="Downloading source...")
    src = _resolve_in(body)
    wanted = [s for s in (body.get("stems") or []) if s in STEM_NAMES]
    if DEMUCS_PYTHON:
        return _demucs_separate(task_id, src, workdir, wanted)

    _progress(task_id, stage="Submitting to StemDeck...")
    job_id = sd_submit(src)
    _progress(task_id, jobId=job_id, stage="Separating stems...")

    deadline = time.time() + JOB_TIMEOUT
    while True:
        info = sd_job(job_id)
        if info["status"] == "done":
            break
        if info["status"] in ("error", "cancelled", "stopped"):
            raise RuntimeError(f"StemDeck 任务 {job_id} 结束于 {info['status']}：{info.get('error') or ''}")
        if time.time() > deadline:
            raise RuntimeError(f"StemDeck 任务 {job_id} 超过 {JOB_TIMEOUT} 秒未完成")
        _progress(task_id, stage=info.get("stage"), progress=info.get("progress"))
        time.sleep(POLL_SECONDS)

    _progress(task_id, stage="Fetching stems...")
    stems, zip_path = _collect(task_id, job_id, wanted, workdir)
    return {
        "ok": True,
        "taskId": task_id,
        "status": "done",
        "operation": "separate",
        "jobId": job_id,
        "duration": info.get("duration"),
        "stems": stems,
        "zip": {"bytes": _bytes(zip_path), "url": f"/result/{task_id}/all.zip"},
        "__upload": (zip_path, "application/zip"),
    }


def run_task(task_id, body):
    workdir = os.path.join(tempfile.gettempdir(), "dsp-" + task_id)
    os.makedirs(workdir, exist_ok=True)
    started = time.time()
    try:
        cls = engines.get(body.get("module") or "")
        if cls:
            result = _run_local(task_id, body, workdir, cls)
        else:
            result = _run_separate(task_id, body, workdir)

        result["seconds"] = round(time.time() - started, 2)
        result["deduplicated"] = False
        if OSS_BUCKET and result.get("operation") == "separate":
            result["oss"] = _publish_oss(task_id, workdir, result)
        upload = result.pop("__upload", None)
        if body.get("uploadUrl") and upload:
            result["uploadStatus"] = _upload(upload[0], body["uploadUrl"], body.get("uploadMethod"),
                                             upload[1])
            result["uploadUrl"] = body["uploadUrl"]
        if OSS_BUCKET:
            shutil.rmtree(workdir, ignore_errors=True)
        _progress(task_id, status="done", stage="Done", result=result, finished=time.time())
        return result
    except Exception as e:
        sys.stderr.write(traceback.format_exc())
        _progress(task_id, status="failed", error=str(e), finished=time.time())
        raise


def _view(task_id):
    job = _jobs.get(task_id)
    if not job:
        return None
    view = {k: job[k] for k in ("taskId", "status", "stage", "progress", "jobId", "created") if k in job}
    for extra in ("result", "error"):
        if extra in job:
            view[extra] = job[extra]
    if "finished" in job:
        view["elapsed"] = round(job["finished"] - job["created"], 2)
    return {"ok": True, **view}


def _start(body):
    task_id = str(body.get("taskId") or _next_id())
    with _lock:
        existing = _jobs.get(task_id)
        if existing and existing["status"] in ("done", "failed"):
            view = _view(task_id)
            if "result" in view:
                view["result"] = {**view["result"], "deduplicated": True}
            return view
        if existing:
            return _view(task_id)
        _jobs[task_id] = {"taskId": task_id, "status": "queued", "created": time.time()}
    if body.get("async"):
        threading.Thread(target=_run_quiet, args=(task_id, body), daemon=True).start()
        return _view(task_id)
    return run_task(task_id, body)


def _run_quiet(task_id, body):
    try:
        run_task(task_id, body)
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, data):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _file(self, path, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        # 有的客户端写文件会带 BOM，严格 utf-8 会直接拒掉
        return json.loads(self.rfile.read(n).decode("utf-8-sig"))

    def _receive(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            raise ValueError("空请求体")
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        ext = ctype.split("/")[-1]
        if not re.fullmatch(r"[a-z0-9]{1,8}", ext) or ext in ("octet-stream", "json", "plain"):
            raise ValueError(f"Content-Type 需为音频类型，例如 audio/mpeg，当前 {ctype or '未设置'}")
        ext = _UPLOAD_EXT.get(ext, ext)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_id = uuid.uuid4().hex
        dest = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
        left = n
        with open(dest, "wb") as f:
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        info = sf.info(dest)
        return {"ok": True, "fileId": file_id, "bytes": n - left, "format": ext,
                "duration": round(info.frames / info.samplerate, 3)}

    def _authed(self):
        if not TOKEN:
            return True
        header = self.headers.get("Authorization") or ""
        if header.lower().startswith("bearer "):
            return header[7:].strip() == TOKEN
        return (self.headers.get("X-DSP-Token") or "") == TOKEN

    def do_GET(self):
        path = self.path.split("?")[0]
        # 文档和健康检查对外公开，方便对面先读接口再拿 token
        if not (path == "/health" or path.startswith("/api/docs")) and not self._authed():
            return self._json(401, {"ok": False, "error": "missing or wrong token"})
        if path.startswith("/api/docs"):
            raw = DOCS.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            return self.wfile.write(raw)
        if path == "/capabilities":
            return self._json(200, capabilities())
        if path == "/health":
            if DEMUCS_PYTHON:
                out = {"ok": True, "engine": "demucs",
                       "engine_ok": os.path.isfile(DEMUCS_PYTHON),
                       "model": DEMUCS_MODEL, "device": DEMUCS_DEVICE}
            else:
                out = {"ok": True, "engine": "stemdeck", "engine_ok": False}
                try:
                    h = sd_health()
                    out.update(engine_ok=True, version=h.get("version"),
                               model=h.get("demucs_model"), device=h.get("demucs_device"))
                except Exception as e:
                    out["engine_error"] = str(e)
            with _lock:
                out["tasks"] = len(_jobs)
            return self._json(200, out)
        if path.startswith("/task/"):
            view = _view(path[len("/task/"):])
            if view is None:
                return self._json(404, {"ok": False, "error": f"unknown task {path[6:]}"})
            return self._json(200, view)
        if path == "/tasks":
            with _lock:
                views = [_view(k) for k in list(_jobs)[-50:]]
            return self._json(200, {"ok": True, "tasks": [v for v in reversed(views) if v]})
        if path.startswith("/result/"):
            parts = path[len("/result/"):].split("/")
            if len(parts) != 2:
                return self._json(404, {"ok": False, "error": "bad result path"})
            task_id, name = parts
            if task_id not in _jobs:
                return self._json(404, {"ok": False, "error": f"unknown task {task_id}"})
            workdir = os.path.join(tempfile.gettempdir(), "dsp-" + task_id)
            if name == "all.zip":
                target, ctype = os.path.join(workdir, "all.zip"), "application/zip"
            elif name == "out.wav":
                target, ctype = os.path.join(workdir, "out.wav"), "audio/wav"
            elif re.fullmatch(r"[a-z]+\.wav", name):
                stem = _stem_key(name)
                hits = [f for f in os.listdir(workdir)
                        if f.lower().endswith(".wav") and _stem_key(f) == stem] if os.path.isdir(workdir) else []
                if not hits:
                    return self._json(404, {"ok": False, "error": f"no stem {name}"})
                target, ctype = os.path.join(workdir, hits[0]), "audio/wav"
            else:
                return self._json(404, {"ok": False, "error": "bad result name"})
            if not os.path.isfile(target):
                return self._json(404, {"ok": False, "error": "result not ready"})
            return self._file(target, ctype)
        self._json(404, {"ok": False, "error": f"unknown path {path}"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/process", "/separate", "/upload", "/warmup"):
            return self._json(404, {"ok": False, "error": f"unknown path {path}"})
        if not self._authed():
            return self._json(401, {"ok": False, "error": "missing or wrong token"})
        try:
            if path == "/warmup":
                return self._json(200, _warmup())
            self._json(200, self._receive() if path == "/upload" else _start(self._body()))
        except Exception as e:
            sys.stderr.write(traceback.format_exc())
            self._json(400, {"ok": False, "error": str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    engine = f"demucs {DEMUCS_MODEL} on {DEMUCS_DEVICE}" if DEMUCS_PYTHON else STEMDECK
    sys.stderr.write(f"local dsp server on http://{HOST}:{PORT} -> {engine}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
