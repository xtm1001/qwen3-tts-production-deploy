# ==============================================================================
# Qwen3-TTS Dual-Mode API Service
# 
# 生产级 Qwen3-TTS 语音合成 API 服务。
# 支持 CustomVoice, VoiceDesign, Base 克隆三种模式。
# 提供 Voice ID 加载、长文本智能分片、高级采样参数控制等功能。
#
# License: MIT
# Author: 小天魔 (xtm1001)
# Repository: https://github.com/xtm1001/qwen3-tts-production-deploy
# ==============================================================================

import os
import io
import re
import time
import uuid
import threading
import pickle
import torch
from contextlib import contextmanager
import soundfile as sf
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

# ============ 配置 ============
APP_VERSION = "1.6.0"
VOICE_ID_DIR = os.getenv("VOICE_ID_DIR", "/app/reference_audio")

MODEL_KEY = os.getenv("MODEL_KEY", "")
MODEL_MAP = {
    "custom_1.7b":  "/data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design_1.7b":  "/data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base_1.7b":    "/data/models/Qwen3-TTS-12Hz-1.7B-Base",
    "custom_0.6b":  "/data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base_0.6b":    "/data/models/Qwen3-TTS-12Hz-0.6B-Base",
}

app = FastAPI(title="Qwen3-TTS Dual-Mode", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 统计收集器 ============
class StatsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.total_inference_time = 0.0
        self.last_error = None

    def record_start(self):
        with self._lock: self.total_requests += 1

    def record_success(self, elapsed: float):
        with self._lock:
            self.success_requests += 1
            self.total_inference_time += elapsed

    def record_failure(self, error: str):
        with self._lock:
            self.failed_requests += 1
            self.last_error = error

    def snapshot(self):
        with self._lock:
            avg = (self.total_inference_time / self.success_requests) if self.success_requests > 0 else 0
            return {
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "failed_requests": self.failed_requests,
                "avg_inference_sec": round(avg, 2),
                "last_error": self.last_error,
            }

stats = StatsCollector()

# ============ 工具函数 ============
def split_for_tts(text: str, max_chars: int = 400, min_chars: int = 20):
    text = text.strip()
    parts = re.split(r'(?<=[。！？!?\n])', text)
    chunks, cur = [], ""
    for p in parts:
        p = p.strip()
        if not p: continue
        if len(cur) + len(p) <= max_chars: cur += p
        else:
            if cur: chunks.append(cur)
            while len(p) > max_chars:
                cut = p[:max_chars]
                m = re.search(r'[，；,;]', cut[::-1])
                if m:
                    idx = len(cut) - m.start()
                    chunks.append(p[:idx])
                    p = p[idx:]
                else:
                    chunks.append(cut)
                    p = p[max_chars:]
            cur = p
    if cur: chunks.append(cur)
    out = []
    for c in chunks:
        if out and len(out[-1]) < min_chars: out[-1] += c
        else: out.append(c)
    return [c for c in out if c.strip()]

def concat_wavs(wav_list, sr, silence_sec=0.15, head_padding_sec=0.0):
    if not wav_list: return None, sr
    out = []
    if head_padding_sec > 0:
        out.append(np.zeros(int(sr * head_padding_sec), dtype=wav_list[0].dtype))
    sil = np.zeros(int(sr * silence_sec), dtype=wav_list[0].dtype)
    for i, w in enumerate(wav_list):
        out.append(w)
        if i != len(wav_list) - 1: out.append(sil)
    return np.concatenate(out, axis=0), sr

def _clean_text_front(text: str) -> str:
    text = re.sub(r'^[啊呀哎唉哦噢喔哟呦嗯唔呃嘿哈哼呸嗬]', '', text)
    text = re.sub(r'^[(（][^)）][)）]', '', text)
    text = re.sub(r'^[,.，。、：:;；]+', '', text)
    return text.strip()

# ============ 模式 A：模型服务 ============
if MODEL_KEY:
    from qwen_tts import Qwen3TTSModel
    
    model = None
    inference_lock = threading.Lock()
    DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    _dtype_env = os.getenv("DTYPE", "").lower()
    DTYPE = "bfloat16" if _dtype_env in ("bf16", "bfloat16") else ("float16" if _dtype_env in ("fp16", "float16") else ("float32" if _dtype_env in ("fp32", "float32") else ("bfloat16" if torch.cuda.is_available() else "float32")))
    MAX_INFERENCE_SEC = int(os.getenv("MAX_INFERENCE_SEC", "0"))

    @app.on_event("startup")
    def load_model():
        global model
        if MODEL_KEY not in MODEL_MAP: raise RuntimeError(f"未知 MODEL_KEY={MODEL_KEY}")
        model_path = os.path.normpath(os.path.abspath(MODEL_MAP[MODEL_KEY]))
        print(f"🚀 加载模型: {MODEL_KEY} | 路径: {model_path} | 设备: {DEVICE} | 精度: {DTYPE}", flush=True)
        
        load_kwargs = {"device_map": DEVICE, "dtype": DTYPE}
        attn_env = os.getenv("ATTN", "auto").lower()
        if attn_env == "flash_attention_2":
            try:
                import flash_attn
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError: load_kwargs["attn_implementation"] = "sdpa"
        else:
            try:
                import flash_attn
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError: load_kwargs["attn_implementation"] = "sdpa"
            
        if os.path.exists(model_path): load_kwargs["local_files_only"] = True
        else: raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        try:
            model = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
            print(f"✅ 模型 {MODEL_KEY} 加载完成！", flush=True)
        except Exception as e:
            print(f"❌ 模型加载失败: {e}", flush=True)
            raise

    class TTSRequest(BaseModel):
        text: Optional[str] = None
        input: Optional[str] = None
        voice: Optional[str] = "Vivian"
        speaker: Optional[str] = None
        instruct: Optional[str] = None
        language: Optional[str] = "Chinese"
        ref_audio: Optional[str] = None
        ref_text: Optional[str] = None
        voice_id: Optional[str] = None
        response_format: str = "wav"
        seed: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None
        repetition_penalty: Optional[float] = None

    @app.get("/health")
    def health(): return {"status": "ok" if model else "loading", "mode": "model-server", "model": MODEL_KEY, "version": APP_VERSION}
    
    @app.get("/ready")
    def ready(): return {"ready": model is not None}
    
    @app.get("/stats")
    def get_stats(): return stats.snapshot()

    def _norm_text(req: TTSRequest) -> str:
        t = req.text or req.input
        if not t or not str(t).strip(): raise HTTPException(422, "缺少文本")
        return _clean_text_front(t.strip())

    @contextmanager
    def request_timer():
        req_id = str(uuid.uuid4())[:8]
        start = time.time()
        print(f"[{req_id}]  请求开始", flush=True)
        try: yield req_id, start
        finally:
            elapsed = time.time() - start
            print(f"[{req_id}] 📤 完成，耗时 {elapsed:.1f}s", flush=True)

    @app.post("/v1/audio/speech")
    async def generate(request: Request, req: TTSRequest):
        if not model: raise HTTPException(503, "模型未就绪")
        text = _norm_text(req)
        speaker = (req.speaker or req.voice or "Vivian").strip()
        lang = (req.language or "Chinese").strip()
        stats.record_start()
        
        with request_timer() as (req_id, start_time):
            try:
                # Custom 模式
                if "custom" in MODEL_KEY:
                    max_chars = int(os.getenv("TTS_MAX_CHARS", "400"))
                    chunks = split_for_tts(text, max_chars=max_chars) if len(text) > max_chars else [text]
                    waves = []
                    if not inference_lock.acquire(blocking=False): raise HTTPException(429, "GPU 忙")
                    try:
                        for i, ch in enumerate(chunks):
                            if await request.is_disconnected(): raise HTTPException(499, "客户端断开")
                            gen_kwargs = {"text": ch, "language": lang, "speaker": speaker, "instruct": req.instruct}
                            if req.seed is not None: gen_kwargs["seed"] = req.seed
                            if req.temperature is not None: gen_kwargs["temperature"] = req.temperature
                            if req.top_p is not None: gen_kwargs["top_p"] = req.top_p
                            if req.top_k is not None: gen_kwargs["top_k"] = req.top_k
                            if req.repetition_penalty is not None: gen_kwargs["repetition_penalty"] = req.repetition_penalty
                            wavs, sr = model.generate_custom_voice(**gen_kwargs)
                            if wavs: waves.append(np.asarray(wavs[0]))
                    finally: inference_lock.release()
                    if not waves: raise HTTPException(500, "空音频")
                    audio, sr = concat_wavs(waves, sr)
                    buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV", subtype="PCM_16"); buf.seek(0)
                    stats.record_success(time.time() - start_time)
                    headers = {"Content-Disposition": "attachment; filename=speech.wav"}
                    if req.seed is not None: headers["X-Used-Seed"] = str(req.seed)
                    return StreamingResponse(buf, media_type="audio/wav", headers=headers)

                # Design 模式
                elif "design" in MODEL_KEY:
                    if not req.instruct: raise HTTPException(400, "需要 instruct")
                    max_chars = int(os.getenv("TTS_MAX_CHARS", "400"))
                    chunks = split_for_tts(text, max_chars=max_chars) if len(text) > max_chars else [text]
                    waves = []
                    if not inference_lock.acquire(blocking=False): raise HTTPException(429, "GPU 忙")
                    try:
                        for i, ch in enumerate(chunks):
                            if await request.is_disconnected(): raise HTTPException(499, "客户端断开")
                            gen_kwargs = {"text": ch, "language": lang, "instruct": req.instruct}
                            if req.seed is not None: gen_kwargs["seed"] = req.seed
                            if req.temperature is not None: gen_kwargs["temperature"] = req.temperature
                            if req.top_p is not None: gen_kwargs["top_p"] = req.top_p
                            if req.top_k is not None: gen_kwargs["top_k"] = req.top_k
                            if req.repetition_penalty is not None: gen_kwargs["repetition_penalty"] = req.repetition_penalty
                            wavs, sr = model.generate_voice_design(**gen_kwargs)
                            if wavs: waves.append(np.asarray(wavs[0]))
                    finally: inference_lock.release()
                    if not waves: raise HTTPException(500, "空音频")
                    audio, sr = concat_wavs(waves, sr)
                    buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV", subtype="PCM_16"); buf.seek(0)
                    stats.record_success(time.time() - start_time)
                    headers = {"Content-Disposition": "attachment; filename=speech.wav"}
                    if req.seed is not None: headers["X-Used-Seed"] = str(req.seed)
                    return StreamingResponse(buf, media_type="audio/wav", headers=headers)

                # Base 克隆模式
                elif "base" in MODEL_KEY:
                    prompt_items = None
                    if req.voice_id:
                        voice_id_basename = os.path.basename(req.voice_id)
                        if voice_id_basename.endswith('.pkl'):
                            pkl_path = os.path.join(VOICE_ID_DIR, voice_id_basename)
                        else:
                            pkl_path = os.path.join(VOICE_ID_DIR, f"{voice_id_basename}.pkl")
                        if os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 100:
                            print(f"[{req_id}] 加载预存 Voice ID: {pkl_path}", flush=True)
                            try:
                                prompt_items = torch.load(pkl_path, map_location=DEVICE, weights_only=False)
                            except Exception:
                                with open(pkl_path, 'rb') as f:
                                    prompt_items = pickle.load(f)
                        else:
                            raise HTTPException(404, f"Voice ID 文件不存在或已损坏: {pkl_path}")
                    elif req.ref_audio:
                        print(f"[{req_id}] 实时提取 Voice Prompt...", flush=True)
                        try:
                            prompt_items = model.create_voice_clone_prompt(
                                ref_audio=req.ref_audio,
                                ref_text=req.ref_text or "",
                                x_vector_only_mode=False
                            )
                        except Exception as e:
                            safe_err = str(e).encode('utf-8', errors='ignore').decode('utf-8')
                            raise HTTPException(500, f"提取 Prompt 失败: {safe_err}")
                    else:
                        raise HTTPException(400, "Base 模式需要提供 ref_audio 或 voice_id")

                    max_chars = int(os.getenv("TTS_MAX_CHARS", "400"))
                    chunks = split_for_tts(text, max_chars=max_chars) if len(text) > max_chars else [text]
                    waves = []
                    if not inference_lock.acquire(blocking=False): raise HTTPException(429, "GPU 忙")
                    try:
                        for i, ch in enumerate(chunks):
                            if await request.is_disconnected(): raise HTTPException(499, "客户端断开")
                            clone_kwargs = {"text": ch, "language": lang, "voice_clone_prompt": prompt_items}
                            if req.seed is not None: clone_kwargs["seed"] = req.seed
                            if req.temperature is not None: clone_kwargs["temperature"] = req.temperature
                            if req.top_p is not None: clone_kwargs["top_p"] = req.top_p
                            if req.top_k is not None: clone_kwargs["top_k"] = req.top_k
                            if req.repetition_penalty is not None: clone_kwargs["repetition_penalty"] = req.repetition_penalty
                            wavs, sr = model.generate_voice_clone(**clone_kwargs)
                            if wavs: waves.append(np.asarray(wavs[0]))
                    finally: inference_lock.release()
                    if not waves: raise HTTPException(500, "空音频")
                    audio, sr = concat_wavs(waves, sr)
                    buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV", subtype="PCM_16"); buf.seek(0)
                    stats.record_success(time.time() - start_time)
                    headers = {"Content-Disposition": "attachment; filename=speech.wav"}
                    if req.seed is not None: headers["X-Used-Seed"] = str(req.seed)
                    return StreamingResponse(buf, media_type="audio/wav", headers=headers)
                else:
                    raise HTTPException(400, f"不支持的 MODEL_KEY: {MODEL_KEY}")
                    
            except HTTPException: raise
            except Exception as e:
                stats.record_failure(str(e))
                safe_err = str(e).encode('utf-8', errors='ignore').decode('utf-8')
                print(f"[{req_id}]  推理失败: {safe_err}", flush=True)
                raise HTTPException(500, f"推理失败: {safe_err}")

# ============ 模式 B：Backend 转发 ============
else:
    endpoints_raw = os.getenv("MODEL_ENDPOINTS", "")
    MODEL_ENDPOINTS = {}
    if endpoints_raw:
        for pair in endpoints_raw.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                MODEL_ENDPOINTS[k.strip()] = v.strip()
    else:
        MODEL_ENDPOINTS = {"custom": "http://127.0.0.1:18880", "design": "http://127.0.0.1:18881", "base": "http://127.0.0.1:18900"}

    class TTSRequest(BaseModel):
        text: Optional[str] = None
        input: Optional[str] = None
        model_type: str = "custom"
        voice: Optional[str] = "Vivian"
        speaker: Optional[str] = None
        instruct: Optional[str] = None
        language: Optional[str] = "Chinese"
        ref_audio: Optional[str] = None
        ref_text: Optional[str] = None
        voice_id: Optional[str] = None
        seed: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None
        repetition_penalty: Optional[float] = None

    @app.get("/health")
    def health(): return {"status": "ok", "mode": "backend-proxy", "endpoints": MODEL_ENDPOINTS}
    
    @app.post("/v1/audio/speech")
    async def proxy_tts(req: TTSRequest):
        endpoint = MODEL_ENDPOINTS.get(req.model_type)
        if not endpoint: raise HTTPException(400, f"未知 model_type: {req.model_type}")
        text = req.text or req.input
        if not text or not str(text).strip(): raise HTTPException(422, "缺少文本")
        
        payload = {"text": text, "language": req.language or "Chinese"}
        if req.model_type == "custom":
            payload["speaker"] = req.speaker or req.voice or "Vivian"
            if req.instruct: payload["instruct"] = req.instruct
        elif req.model_type == "design":
            if not req.instruct: raise HTTPException(400, "需要 instruct")
            payload["instruct"] = req.instruct
        elif req.model_type == "base":
            if req.voice_id: payload["voice_id"] = req.voice_id
            elif req.ref_audio:
                payload["ref_audio"] = req.ref_audio
                if req.ref_text: payload["ref_text"] = req.ref_text
            else:
                raise HTTPException(400, "base 模式需要 ref_audio 或 voice_id")
                
        if req.seed is not None: payload["seed"] = req.seed
        if req.temperature is not None: payload["temperature"] = req.temperature
        if req.top_p is not None: payload["top_p"] = req.top_p
        if req.top_k is not None: payload["top_k"] = req.top_k
        if req.repetition_penalty is not None: payload["repetition_penalty"] = req.repetition_penalty
                
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(f"{endpoint}/v1/audio/speech", json=payload, timeout=600)
                if resp.status_code != 200: raise HTTPException(resp.status_code, resp.text)
                headers = {"Content-Disposition": "attachment; filename=speech.wav"}
                if req.seed is not None: headers["X-Used-Seed"] = str(req.seed)
                return StreamingResponse(io.BytesIO(resp.content), media_type="audio/wav", headers=headers)
            except httpx.ConnectError:
                raise HTTPException(503, f"无法连接 {endpoint}")
