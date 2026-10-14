"""녹음 한 개(테이크)의 품질 검사: 앞뒤 무음 자르기, 음량·찢어짐·잡음·말 속도, 받아쓰기로 대본과 일치하는지.

결과 verdict: good(그대로 사용) / warn(써도 되지만 다시 녹음 권장) / bad(학습에서 뺌)
"""
import io
import os
import re

import numpy as np
import requests
import soundfile as sf

from settings import ASR_KEY, ASR_URL  # 비어 있으면 받아쓰기 검사만 건너뛴다

HANGUL = re.compile(r"[가-힣]")
DIGITS = "영일이삼사오육칠팔구"


def read_wav(data: bytes):
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def frames_db(x, sr, win=0.02):
    n = max(1, int(sr * win))
    m = len(x) // n
    if m == 0:
        return np.array([-120.0]), n
    f = x[: m * n].reshape(m, n)
    rms = np.sqrt((f ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9), n


def num_to_ko(m):
    """받아쓰기가 숫자로 적은 부분을 한자어 읽기로(대본은 한글로 썼다). 완벽하진 않아도 오류율을 덜 부풀린다."""
    s = m.group(0)
    try:
        v = int(s)
    except ValueError:
        return s
    if v == 0:
        return "영"
    out = ""
    for unit, name in ((10000, "만"), (1000, "천"), (100, "백"), (10, "십")):
        q, v = divmod(v, unit)
        if q:
            out += ("" if q == 1 else DIGITS[q]) + name
    if v:
        out += DIGITS[v]
    return out


def norm_text(s):
    s = re.sub(r"\d+", num_to_ko, s)
    return "".join(HANGUL.findall(s))


def cer(ref, hyp):
    r, h = norm_text(ref), norm_text(hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def transcribe(wav_bytes):
    if not ASR_URL:
        return None
    try:
        r = requests.post(ASR_URL, headers={"Authorization": f"Bearer {ASR_KEY}"} if ASR_KEY else {},
                          files={"file": ("take.wav", wav_bytes, "audio/wav")}, data={"language": "ko"}, timeout=30)
        if r.ok:
            return r.json().get("text", "")
    except requests.RequestException:
        pass
    return None


def analyze(data: bytes, text: str, group: str, use_asr=True):
    """테이크 검사. 반환: (결과 dict, 잘라 낸 wav bytes 또는 None)"""
    try:
        x, sr = read_wav(data)
    except Exception:  # noqa: BLE001
        return {"verdict": "bad", "issues": ["녹음 파일을 읽을 수 없어요"]}, None
    issues, warns = [], []
    if sr < 22050:
        return {"verdict": "bad", "issues": [f"샘플레이트가 너무 낮아요({sr}Hz)"]}, None
    db, n = frames_db(x, sr)
    noise = float(np.percentile(db, 10))
    active = db > max(noise + 12, -55)
    if not active.any():
        return {"verdict": "bad", "issues": ["목소리가 거의 들리지 않아요"], "noise_db": round(noise, 1)}, None
    idx = np.nonzero(active)[0]
    pad = int(0.2 * sr)
    a, b = max(0, idx[0] * n - pad), min(len(x), (idx[-1] + 1) * n + pad)
    y = x[a:b]
    speech_db = float(np.mean(db[active]))
    speech_s = float(active.sum() * n / sr)
    clip = float(np.mean(np.abs(x) >= 0.995))
    snr = speech_db - noise
    syl = len(HANGUL.findall(text))
    rate = syl / max(0.3, (b - a) / sr - 0.4)

    if clip > 0.002:
        issues.append("소리가 너무 커서 찢어졌어요. 마이크에서 조금 떨어져 주세요")
    elif clip > 0.0003:
        warns.append("살짝 찢어지는 부분이 있어요")
    if speech_db < -42:
        issues.append("목소리가 너무 작아요. 마이크에 가까이 와 주세요")
    elif speech_db < -34:
        warns.append("목소리가 조금 작아요")
    if snr < 15:
        issues.append("주변 소음이 커요. 조용한 곳에서 녹음해 주세요")
    elif snr < 25:
        warns.append("주변 소음이 조금 있어요")
    if group != "reaction" and syl >= 6:
        if rate > 10:
            issues.append("문장이 잘린 것 같아요. 끝까지 읽은 뒤 멈춰 주세요")
        elif rate < 2.0:
            warns.append("중간에 쉬는 시간이 길어요")
    if (b - a) / sr > 15:
        warns.append("녹음이 너무 길어요(15초 넘음)")

    # 앞뒤를 자르고, 말소리 평균이 -20dBFS 근처가 되게 맞춘다(봉우리는 -1dBFS를 넘지 않게)
    gain = 10 ** ((-20 - speech_db) / 20)
    peak = float(np.max(np.abs(y))) or 1.0
    gain = min(gain, 0.89 / peak)
    y = np.clip(y * gain, -1, 1)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    out = buf.getvalue()

    hyp, c = None, None
    if use_asr and not issues and group != "reaction":
        hyp = transcribe(out)
        if hyp is not None:
            c = cer(text, hyp)
            if c > 0.35:
                issues.append("대본과 다르게 들려요. 대본 그대로 읽어 주세요")
            elif c > 0.15:
                warns.append("대본과 조금 다르게 들려요")
    verdict = "bad" if issues else ("warn" if warns else "good")
    res = {"verdict": verdict, "issues": issues + warns, "duration": round((b - a) / sr, 2), "speech_s": round(speech_s, 2),
           "speech_db": round(speech_db, 1), "noise_db": round(noise, 1), "snr": round(snr, 1), "clip": round(clip, 5),
           "rate": round(rate, 1), "sr": sr, "asr": hyp, "cer": None if c is None else round(c, 3)}
    return res, out  # 나쁜 테이크도 파일은 남긴다(사용자가 들어 보고 다시 녹음할 수 있게). 학습에서는 뺀다
