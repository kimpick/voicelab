"""합성: 사용자가 학습한 모델(GPT+SoVITS)을 올려 두고, 고른 톤(style)의 본인 녹음을 레퍼런스로 써서 읽는다.

GPU 메모리(8GB, 다른 서버와 같이 쓸 수 있음) 때문에 모델은 한 번에 한 사람 것만 올린다(마지막 사용 기준으로 교체).
가속(GPT 토큰 생성 CUDA Graph, SoVITS 길이 구간화)을 그대로 쓴다.
"""
import gc
import io
import os
import sys
import threading
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gsv"))
from tts_engine import load_tts  # noqa: E402
_lock = threading.Lock()
_cur = {"key": None, "tts": None, "used": 0.0}


def unload():
    with _lock:
        _cur.update(key=None, tts=None)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _get(gpt, sovits):
    key = (gpt, sovits)
    if _cur["key"] != key:
        _cur.update(key=None, tts=None)
        gc.collect()
        tts = load_tts(gpt, sovits)
        try:
            import tts_speedups
            tts_speedups.apply(tts, bucket=True)
            from fast_t2s import FastT2S
            FastT2S(tts.t2s_model.model).install()
        except Exception as e:  # noqa: BLE001  가속 실패해도 원래 엔진으로 읽는다
            print("voicelab synth: speedups off:", e)
        _cur.update(key=key, tts=tts)
    _cur["used"] = time.time()
    return _cur["tts"]


def synth(gpt, sovits, text, ref_path, ref_text, aux=(), speed=1.0, temperature=1.0, seed=None, fmt="mp3"):
    with _lock:
        tts = _get(gpt, sovits)
        sr, audio = next(tts.run({
            "text": text, "text_lang": "ko",
            "ref_audio_path": ref_path, "prompt_text": ref_text, "prompt_lang": "ko",
            "aux_ref_audio_paths": list(aux),
            "top_k": 15, "top_p": 1, "temperature": temperature,
            "text_split_method": "cut5", "batch_size": 1, "speed_factor": speed,
            "repetition_penalty": 1.35, "seed": -1 if seed is None else seed,
        }))
    audio = np.asarray(audio)
    pcm = audio if audio.dtype == np.int16 else (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    if fmt == "mp3":
        sf.write(buf, pcm, sr, format="MP3", subtype="MPEG_LAYER_III")
        return buf.getvalue(), "audio/mpeg"
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


def idle_unload(after=900):
    """15분 안 쓰면 모델을 내려 GPU를 비워 둔다(다른 서버·학습용)."""
    while True:
        time.sleep(60)
        if _cur["tts"] is not None and time.time() - _cur["used"] > after:
            unload()


threading.Thread(target=idle_unload, daemon=True, name="voicelab-unload").start()
