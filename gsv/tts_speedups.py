"""GPT-SoVITS 실시간용 가속 모음.

1) 매 요청 끝의 메모리 정리(gc + cuda.empty_cache, 약 0.21초)를 EMPTY_EVERY번에 한 번만 한다(tts_profile.py, 10-07).
2) v2Pro 계열이 요청마다 다시 계산하는 레퍼런스 음성 임베딩(약 0.12초)을 레퍼런스별로 기억해 둔다(10-07).
3) SoVITS 디코딩 길이 구간화(10-08): 디코딩은 의미 토큰 길이가 처음 보는 값일 때마다 cuDNN 준비로 약 0.3초를
   더 쓴다(같은 길이면 0.1초, scratchpad newshape 실험). 토큰 길이를 BUCKET 단위로 올림해 뒤를 마스크로 가리고
   디코딩한 뒤 원래 길이로 자른다. 인코더·flow는 마스크를 쓰므로 원래 길이 부분의 계산이 같고, 생성기(dec)만
   끝부분 몇 프레임이 아주 조금 달라질 수 있다(끝은 원래 말끝 무음 쪽).
4) warmup(): 서버 시작 때 구간 길이마다 한 번씩 디코딩해 두어 첫 요청부터 빠르게 한다.
"""
import torch
import torch.nn.functional as F

EMPTY_EVERY = 20
BUCKET = 8          # 의미 토큰 8개 = 0.32초
WARM_MAX_TOKENS = 400  # 약 16초 분량까지 미리 데운다


def apply(tts, bucket=True):
    original_empty = tts.empty_cache
    calls = {"n": 0}

    def empty_cache_sometimes():
        calls["n"] += 1
        if calls["n"] % EMPTY_EVERY == 0:
            original_empty()

    tts.empty_cache = empty_cache_sometimes

    sv = getattr(tts, "sv_model", None)
    if sv is not None:
        compute = sv.compute_embedding3
        cache = {}

        def cached(wav):
            # 같은 레퍼런스면 같은 텐서가 다시 들어온다. 주소, 모양, 앞부분 값 합으로 알아본다
            key = (wav.data_ptr(), tuple(wav.shape), float(wav.flatten()[:4000].sum()))
            if key not in cache:
                if len(cache) > 32:
                    cache.clear()
                with torch.no_grad():
                    cache[key] = compute(wav)
            return cache[key]

        sv.compute_embedding3 = cached
    if bucket:
        bucket_decode(tts.vits_model)
    return tts


def bucket_decode(vits, g=BUCKET):
    """vits.decode를 길이 구간화 버전으로 바꾼다. 원본 SynthesizerTrn.decode(models.py)와 같은 계산에
    y_lengths만 원래 길이로 준다."""
    if getattr(vits, "_bucketed", False):
        return
    orig = vits.decode
    hop = 1
    for r in vits.upsample_rates:
        hop *= r
    frames_per_token = 2 if vits.semantic_frame_rate == "25hz" else 1

    def get_ge(refer, sv_emb):
        refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
        refer_mask = torch.unsqueeze(torch.arange(refer.size(2), device=refer.device) < refer_lengths[0], 0)
        refer_mask = refer_mask.unsqueeze(0).to(refer.dtype)
        if vits.version == "v1":
            ge = vits.ref_enc(refer * refer_mask, refer_mask)
        else:
            ge = vits.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        if vits.is_v2pro:
            ge = ge + vits.sv_emb(sv_emb).unsqueeze(-1)
            ge = vits.prelu(ge)
        return ge

    @torch.no_grad()
    def decode(codes, text, refer, noise_scale=0.5, speed=1, sv_emb=None):
        t = codes.size(2)
        tb = -(-t // g) * g
        if speed != 1 or vits.version not in ("v2", "v2Pro", "v2ProPlus"):
            return orig(codes, text, refer, noise_scale, speed, sv_emb)
        if type(refer) == list:
            ge = torch.stack([get_ge(r, sv_emb[i] if vits.is_v2pro else None) for i, r in enumerate(refer)], 0).mean(0)
        else:
            ge = get_ge(refer, sv_emb)
        codes_p = F.pad(codes, (0, tb - t)) if tb > t else codes
        y_lengths = torch.LongTensor([t * frames_per_token]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)
        quantized = vits.quantizer.decode(codes_p)
        if vits.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
        ge_in = vits.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if vits.is_v2pro else ge
        x, m_p, logs_p, y_mask, _, _ = vits.enc_p(quantized, y_lengths, text, text_lengths, ge_in, speed)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = vits.flow(z_p, y_mask, g=ge, reverse=True)
        o = vits.dec(z * y_mask, g=ge)
        return o[:, :, : t * frames_per_token * hop]

    vits.decode = decode
    vits._bucketed = True


@torch.no_grad()
def warmup(tts, ref, max_tokens=WARM_MAX_TOKENS, log=print):
    """레퍼런스를 한 번 넣고, 구간 길이마다 디코딩을 한 번씩 돌려 cuDNN 준비를 끝내 둔다."""
    import time

    t0 = time.perf_counter()
    next(tts.run({"text": "안녕", "text_lang": "ko", "ref_audio_path": ref["path"], "prompt_text": ref["text"],
                  "prompt_lang": "ko", "top_k": 15, "temperature": 1.0, "text_split_method": "cut5",
                  "batch_size": 1, "seed": 1}))
    vits = tts.vits_model
    dev, dt = tts.configs.device, tts.precision
    refer = [spec.to(dtype=dt, device=dev) for spec, _ in tts.prompt_cache["refer_spec"]]
    sv_emb = [tts.sv_model.compute_embedding3(a) for _, a in tts.prompt_cache["refer_spec"]] if tts.is_v2pro else None
    text = torch.randint(1, 200, (1, 40), device=dev)
    for n in range(BUCKET, max_tokens + 1, BUCKET):
        codes = torch.randint(0, 1024, (1, 1, n), device=dev)
        vits.decode(codes, text, refer, speed=1.0, sv_emb=sv_emb)
    torch.cuda.synchronize()
    log(f"warmup: SoVITS {max_tokens // BUCKET}개 길이 구간 {time.perf_counter() - t0:.1f}s")
