"""GPT(의미 토큰 생성) CUDA Graph 가속 (계획서 Phase 2B: 원본 엔진 직접 가속).

원본은 토큰 하나를 만들 때마다 파이썬이 GPU 명령 수백 개를 하나씩 내린다. i5-9400F에서는 이 명령 발행이
GPU 계산보다 느려서 토큰당 약 23ms가 걸리고 GPU는 30% 정도만 일한다(perf\\baseline).
여기서는 한 토큰을 만드는 계산 전체(24층 트랜스포머 + 반복 억제 + top-k 샘플링 + 다음 입력 준비)를
CUDA Graph 하나로 녹화해 두고, 토큰마다 녹화본을 한 번 재생만 한다.

- 텍스트 전처리, 레퍼런스, SoVITS 디코딩은 원본 그대로다. 한국어 처리도 원본 코드를 그대로 쓴다.
- 처음 한 번(프롬프트 전체를 넣는 단계)은 원본 코드로 계산하고, 그 결과(K/V)를 고정 크기 캐시에 복사한다.
- 캐시 길이는 구간(512/1024/2048)별로 녹화본을 따로 만들어, 짧은 문장은 짧은 구간으로 계산한다.
- 샘플링 규칙은 원본(AR/models/utils.py logits_to_probs, t2s_model.infer_panel_batch_infer)과 같다:
  반복 억제(이미 나온 토큰, 프롬프트 포함) → 처음 11토큰은 끝(EOS) 금지 → 온도 → top-k → 지수분포 샘플링,
  끝 판정은 "뽑힌 토큰이 EOS" 또는 "반복 억제 뒤 최댓값이 EOS".
- 난수 소비 순서가 원본과 달라서 같은 시드라도 똑같은 음성이 나오지는 않는다(분포는 같다).

사용:
    fast = FastT2S(tts.t2s_model.model)
    fast.install()      # TTS.run이 고르는 infer_panel_batch_infer(배치 1)를 이 엔진으로 바꾼다
    fast.uninstall()    # 원래대로
"""
import torch
import torch.nn.functional as F

BUCKETS = (512, 1024, 2048)


class FastT2S:
    def __init__(self, model, buckets=BUCKETS):
        self.m = model
        w0 = model.ar_predict_layer.weight
        self.dev, self.dt = w0.device, w0.dtype
        self.D, self.H = model.model_dim, model.num_head
        self.Dh = self.D // self.H
        self.n_layers = model.num_layers
        self.EOS = model.EOS
        self.V = model.vocab_size
        self.buckets = buckets
        self.max_len = buckets[-1]
        self.layers = []
        for layer in model.h.layers:
            self.layers.append((
                layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias,
                layer.norm1.weight, layer.norm1.bias, layer.norm1.eps,
                layer.linear1.weight, layer.linear1.bias, layer.linear2.weight, layer.linear2.bias,
                layer.norm2.weight, layer.norm2.bias, layer.norm2.eps,
            ))
        self.pred_w = model.ar_predict_layer.weight
        self.emb_w = model.ar_audio_embedding.word_embeddings.weight
        ap = model.ar_audio_position
        ap.extend_pe(torch.zeros(1, 4000, device=self.dev, dtype=self.dt))
        self.pe = ap.pe[0].to(self.dev, self.dt)  # (4000, D)
        self.x_scale = ap.x_scale
        self.alpha = ap.alpha.detach()

        d = dict(device=self.dev)
        B = 1
        self.K = torch.zeros(self.n_layers, B, self.H, self.max_len, self.Dh, dtype=self.dt, **d)
        self.Vc = torch.zeros_like(self.K)
        self.x = torch.zeros(B, 1, self.D, dtype=self.dt, **d)       # 이번에 넣을 토큰의 입력 벡터
        self.pos = torch.zeros(1, dtype=torch.long, **d)              # 캐시에 쓸 위치(= 프롬프트 길이 + 생성 수)
        self.step = torch.zeros(1, dtype=torch.long, **d)             # 지금 뽑는 토큰 번호(원본 idx)
        self.ylen = torch.zeros(1, dtype=torch.long, **d)             # 레퍼런스 토큰 수(위치 임베딩 기준)
        self.seen = torch.zeros(B, self.V, dtype=torch.bool, **d)     # 이미 나온 토큰(반복 억제)
        self.tok = torch.zeros(B, 1, dtype=torch.long, **d)
        self.eos = torch.zeros(B, dtype=torch.bool, **d)
        self.arange = torch.arange(self.max_len, **d)
        self.eos_col = torch.zeros(1, self.V, dtype=torch.bool, **d)
        self.eos_col[0, self.EOS] = True
        # 샘플링 설정도 텐서로 둬서 녹화본을 다시 만들지 않고 바꿀 수 있게 한다(top_k만 녹화 때 고정)
        self.rep = torch.ones(1, dtype=self.dt, **d)
        self.temp = torch.ones(1, dtype=self.dt, **d)
        self.top_k = None
        self.graphs = {}
        self.pool = None
        self._orig = None

    # ---------- 한 토큰 계산(녹화 대상) ----------
    def _sample(self, logits):
        """원본 logits_to_probs + multinomial_sample_one_no_sync와 같은 규칙. 결과를 고정 버퍼에 쓴다."""
        pen = torch.where(logits < 0, logits * self.rep, logits / self.rep)
        logits = torch.where(self.seen, pen, logits)
        logits = logits.masked_fill(self.eos_col & (self.step < 11).view(1, 1), float("-inf"))
        greedy = logits.argmax(dim=-1)
        logits = logits / self.temp
        v, _ = torch.topk(logits, self.top_k)
        logits = torch.where(logits < v[:, -1:], torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        q = torch.empty_like(probs).exponential_(1)
        tok = torch.argmax(probs / q, dim=-1, keepdim=True)
        self.tok.copy_(tok)
        self.eos.copy_((tok[:, 0] == self.EOS) | (greedy == self.EOS))
        self.seen.scatter_(1, tok, True)
        # 다음 입력: 원본 update next step과 같다(위치 = 레퍼런스 길이 + 이번 토큰 번호)
        pe = self.pe.index_select(0, self.ylen + self.step)
        self.x.copy_((F.embedding(tok, self.emb_w) * self.x_scale + self.alpha * pe).to(self.dt))

    def _decode_step(self, n):
        x = self.x
        B = x.shape[0]
        keymask = (self.arange[:n] <= self.pos).view(1, 1, 1, n)
        for i, (qkv_w, qkv_b, out_w, out_b, n1w, n1b, n1e, w1, b1, w2, b2, n2w, n2b, n2e) in enumerate(self.layers):
            q, k, v = F.linear(x, qkv_w, qkv_b).chunk(3, dim=-1)
            q = q.view(B, 1, self.H, self.Dh).transpose(1, 2)
            self.K[i].index_copy_(2, self.pos, k.view(B, 1, self.H, self.Dh).transpose(1, 2))
            self.Vc[i].index_copy_(2, self.pos, v.view(B, 1, self.H, self.Dh).transpose(1, 2))
            a = F.scaled_dot_product_attention(q, self.K[i, :, :, :n], self.Vc[i, :, :, :n], attn_mask=keymask)
            a = F.linear(a.transpose(1, 2).reshape(B, 1, self.D), out_w, out_b)
            x = F.layer_norm(x + a, [self.D], n1w, n1b, n1e)
            x = F.layer_norm(x + F.linear(F.relu(F.linear(x, w1, b1)), w2, b2), [self.D], n2w, n2b, n2e)
        logits = F.linear(x[:, -1], self.pred_w)
        self.step.add_(1)
        self._sample(logits)
        self.pos.add_(1)

    def _graph(self, n):
        if n not in self.graphs:
            # 녹화 전에 옆 스트림에서 몇 번 돌려 둔다(CUDA Graph 규칙). 상태 버퍼는 녹화 뒤 다시 채우므로 상관없다
            saved = [t.clone() for t in (self.x, self.pos, self.step, self.seen, self.tok, self.eos)]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._decode_step(n)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._decode_step(n)
            self.pool = g.pool()
            for t, v in zip((self.x, self.pos, self.step, self.seen, self.tok, self.eos), saved):
                t.copy_(v)
            self.graphs[n] = g
        return self.graphs[n]

    # ---------- 바깥에서 부르는 부분 ----------
    @torch.no_grad()
    def prefill(self, x_ids, bert, prompt, top_k, temperature, repetition_penalty):
        """원본 infer_panel_batch_infer의 첫 단계(배치 1). 끝나면 self.tok에 첫 토큰이 있다."""
        m = self.m
        if self.top_k != top_k:
            self.top_k = top_k
            self.graphs.clear()  # top_k는 녹화본에 박혀 있다
        self.rep.fill_(repetition_penalty)
        self.temp.fill_(max(temperature, 1e-5))
        x = m.ar_text_embedding(x_ids.unsqueeze(0))
        x = x + m.bert_proj(bert.transpose(0, 1).unsqueeze(0))
        x = m.ar_text_position(x)
        y = prompt
        y_pos = m.ar_audio_position(m.ar_audio_embedding(y))
        xy_pos = torch.concat([x, y_pos], dim=1)
        x_len, y_len = x.shape[1], y.shape[1]
        src_len = x_len + y_len
        x_mask = F.pad(torch.zeros(x_len, x_len, dtype=torch.bool, device=self.dev), (0, y_len), value=True)
        y_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=self.dev), diagonal=1),
                       (x_len, 0), value=False)
        attn_mask = torch.concat([x_mask, y_mask], dim=0).view(1, 1, src_len, src_len).expand(1, self.H, -1, -1)
        xy_dec, k_cache, v_cache = m.t2s_transformer.process_prompt(xy_pos, attn_mask, None)
        for i in range(self.n_layers):
            self.K[i, :, :, :src_len] = k_cache[i].view(1, src_len, self.H, self.Dh).transpose(1, 2)
            self.Vc[i, :, :, :src_len] = v_cache[i].view(1, src_len, self.H, self.Dh).transpose(1, 2)
        logits = m.ar_predict_layer(xy_dec[:, -1])
        self.seen.zero_()
        self.seen.scatter_(1, y.long(), True)
        self.step.zero_()
        self.ylen.fill_(y_len)
        self._sample(logits)
        self.pos.fill_(src_len)
        return src_len

    @torch.no_grad()
    def generate(self, x_ids, bert, prompt, top_k=15, temperature=1.0, repetition_penalty=1.35,
                 early_stop_num=-1, chunk=None):
        """의미 토큰을 만든다. chunk가 있으면 chunk개씩 끊어서 내놓고(스트리밍), 없으면 끝에 한 번에 내놓는다.
        내놓는 값: (토큰 텐서 [n], 끝났는지)."""
        src_len = self.prefill(x_ids, bert, prompt, top_k, temperature, repetition_penalty)
        out = []
        pending = []
        idx = 0
        while True:
            # idx번째 토큰이 self.tok에 있다. 끝 판정은 원본처럼 이 토큰을 버리고 멈춘다
            if bool(self.eos[0]):
                break
            pending.append(self.tok.clone())
            idx += 1
            if (early_stop_num != -1 and idx > early_stop_num) or idx >= 1500 or src_len + idx >= self.max_len:
                break
            if chunk and len(pending) >= chunk:
                yield torch.cat(pending, dim=1)[0], False
                out.extend(pending)
                pending = []
            n = next(b for b in self.buckets if b > src_len + idx)
            self._graph(n).replay()
        out.extend(pending)
        if chunk:
            yield (torch.cat(pending, dim=1)[0] if pending else None), True
        else:
            yield torch.cat(out, dim=1)[0], True

    # ---------- TTS 연결 ----------
    def infer_panel_batch_infer(self, x, x_lens, prompts, bert_feature, top_k=-100, top_p=100, early_stop_num=-1,
                                temperature=1.0, repetition_penalty=1.35, **kw):
        if prompts is None or len(x) != 1 or (top_p is not None and top_p < 1):
            return self._orig(x, x_lens, prompts, bert_feature, top_k=top_k, top_p=top_p,
                              early_stop_num=early_stop_num, temperature=temperature,
                              repetition_penalty=repetition_penalty, **kw)
        tokens, _ = next(self.generate(x[0], bert_feature[0], prompts, top_k, temperature, repetition_penalty,
                                       early_stop_num))
        return [tokens], [int(tokens.shape[0])]

    def install(self):
        if self._orig is None:
            self._orig = self.m.infer_panel_batch_infer
            self.m.infer_panel_batch_infer = self.infer_panel_batch_infer
        return self

    def uninstall(self):
        if self._orig is not None:
            self.m.infer_panel_batch_infer = self._orig
            self._orig = None
