"""보이스랩: 회사 사람들이 자기 목소리를 분류별로 녹음하면 개인 TTS를 만들어 주는 웹 서비스.

- 로그인: 구글 계정(config.json의 allowed_domains 회사 도메인, 또는 allowed_emails에 적은 사람만)
- 녹음: 브라우저에서 바로 녹음(파일 올리기는 없음: 남의 목소리를 가져다 쓰는 걸 막기 위해). 테이크마다 자동 품질검사
- 학습: 대기열(그래픽카드 1장). 같은 GPU를 쓰는 다른 서버가 있으면 학습 동안 잠시 내린다(trainer.py)
- 합성: 본인 모델로, 고른 톤의 본인 녹음을 레퍼런스로

사용: start_voicelab.bat  → http://127.0.0.1:8830
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time

import requests
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qc  # noqa: E402
import trainer  # noqa: E402

from settings import DATA, ENV_PATH, PORT  # noqa: E402

USERS = os.path.join(DATA, "users")
COOKIE = "vl_session"
SESSION_DAYS = 14
MIN_NEUTRAL, MIN_TOTAL, MIN_MINUTES = 30, 80, 6.0


def load_config():
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def secret_key():
    """세션 서명 키(.env의 VOICELAB_SECRET). 없으면 만들어 .env에 덧붙인다(화면·기록에 출력하지 않음)."""
    try:
        with open(ENV_PATH, encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("VOICELAB_SECRET="):
                    return ln.split("=", 1)[1].strip().encode()
    except OSError:
        pass
    key = secrets.token_hex(32)
    with open(ENV_PATH, "a", encoding="utf-8") as f:
        f.write(f"\nVOICELAB_SECRET={key}\n")
    return key.encode()


SECRET = secret_key()
CFG = load_config()
with open(os.path.join(HERE, "prompts.json"), encoding="utf-8") as f:
    PROMPTS = json.load(f)
for p in PROMPTS["prompts"]:
    p["id"] = hashlib.sha1(p["text"].encode()).hexdigest()[:10]
PROMPT_BY_ID = {p["id"]: p for p in PROMPTS["prompts"]}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


# ---------- 세션 ----------

def sign(payload: dict) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()
    sig = hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def unsign(token: str):
    try:
        raw, sig = token.rsplit(".", 1)
        if not hmac.compare_digest(sig, hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()):
            return None
        data = json.loads(base64.urlsafe_b64decode(raw.encode()))
        return data if data.get("exp", 0) > time.time() else None
    except Exception:  # noqa: BLE001
        return None


def uid_of(email):
    return hashlib.sha1(email.lower().encode()).hexdigest()[:12]


def allowed(email, hd):
    cfg = load_config()  # 허용 목록은 서버를 끄지 않고 고칠 수 있게 매번 읽는다
    email = email.lower()
    if email in [e.lower() for e in cfg.get("allowed_emails", [])]:
        return True
    domains = [d.lower() for d in cfg.get("allowed_domains", [])]
    return (hd or "").lower() in domains or email.split("@")[-1] in domains


def current(req: Request):
    s = unsign(req.cookies.get(COOKIE, ""))
    if not s or not allowed(s["email"], s.get("hd")):
        return None
    return s


def need_login():
    return JSONResponse({"error": "로그인이 필요해요"}, status_code=401)


def user_dir(uid):
    d = os.path.join(USERS, uid)
    os.makedirs(os.path.join(d, "takes"), exist_ok=True)
    return d


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


_user_locks = {}


def user_lock(uid):
    return _user_locks.setdefault(uid, threading.Lock())


# ---------- 로그인 ----------

@app.get("/api/config")
def config():
    return {"google_client_id": load_config().get("google_client_id", ""), "service": "보이스랩"}


@app.post("/api/login/google")
def login_google(payload: dict):
    cfg = load_config()
    cid = cfg.get("google_client_id", "")
    cred = str(payload.get("credential", ""))
    if not cid or not cred:
        return JSONResponse({"error": "구글 로그인이 설정되지 않았어요"}, status_code=400)
    try:
        r = requests.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": cred}, timeout=10)
        info = r.json() if r.ok else {}
    except requests.RequestException:
        return JSONResponse({"error": "구글 확인 서버에 연결하지 못했어요"}, status_code=502)
    if info.get("aud") != cid or info.get("email_verified") not in ("true", True) \
            or info.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        return JSONResponse({"error": "로그인 정보를 확인하지 못했어요"}, status_code=401)
    email, hd = info.get("email", ""), info.get("hd", "")
    if not allowed(email, hd):
        return JSONResponse({"error": f"{email} 계정은 사용할 수 없어요. 회사 계정으로 로그인하거나 관리자에게 요청해 주세요."},
                            status_code=403)
    sess = {"email": email, "name": info.get("name") or email.split("@")[0], "hd": hd, "uid": uid_of(email),
            "exp": time.time() + SESSION_DAYS * 86400}
    d = user_dir(sess["uid"])
    meta = read_json(os.path.join(d, "meta.json"), {})
    meta.update(email=email, name=sess["name"], last_login=time.time())
    meta.setdefault("created", time.time())
    write_json(os.path.join(d, "meta.json"), meta)
    resp = JSONResponse({"ok": True, "name": sess["name"]})
    resp.set_cookie(COOKIE, sign(sess), max_age=SESSION_DAYS * 86400, httponly=True, secure=True, samesite="lax")
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


@app.get("/api/me")
def me(req: Request):
    s = current(req)
    if not s:
        return need_login()
    meta = read_json(os.path.join(user_dir(s["uid"]), "meta.json"), {})
    return {"email": s["email"], "name": s["name"], "consent": bool(meta.get("consent")),
            "admin": s["email"].lower() in [e.lower() for e in load_config().get("admins", [])]}


# ---------- 대본·동의 ----------

@app.get("/api/prompts")
def prompts(req: Request):
    if not current(req):
        return need_login()
    return {"groups": PROMPTS["groups"], "styles": PROMPTS["styles"],
            "prompts": [{k: p.get(k) for k in ("id", "group", "style", "text", "hint")} for p in PROMPTS["prompts"]]}


CONSENT_TEXT = "저는 제 목소리로 개인 음성 모델을 만드는 데 동의합니다. 이 모델은 제 목소리로만 사용하겠습니다."


@app.post("/api/consent")
async def consent(req: Request, name: str = Form(...), file: UploadFile = File(...)):
    s = current(req)
    if not s:
        return need_login()
    data = await file.read()
    res, out = qc.analyze(data, CONSENT_TEXT, "neutral", use_asr=True)
    if res["verdict"] == "bad":
        return JSONResponse({"error": " / ".join(res["issues"]) or "다시 녹음해 주세요", "qc": res}, status_code=400)
    d = user_dir(s["uid"])
    with open(os.path.join(d, "consent.wav"), "wb") as f:
        f.write(out)
    meta = read_json(os.path.join(d, "meta.json"), {})
    meta["consent"] = {"name": name.strip()[:40], "text": CONSENT_TEXT, "at": time.time(), "asr": res.get("asr")}
    write_json(os.path.join(d, "meta.json"), meta)
    return {"ok": True, "qc": res}


# ---------- 녹음 ----------

def has_consent(uid):
    return bool(read_json(os.path.join(user_dir(uid), "meta.json"), {}).get("consent"))


@app.post("/api/take/{pid}")
async def take(pid: str, req: Request):
    s = current(req)
    if not s:
        return need_login()
    p = PROMPT_BY_ID.get(pid)
    if not p:
        return JSONResponse({"error": "없는 문장"}, status_code=404)
    if not has_consent(s["uid"]):
        return JSONResponse({"error": "먼저 동의 문장을 녹음해 주세요"}, status_code=428)
    data = await req.body()
    if len(data) > 8_000_000:
        return JSONResponse({"error": "녹음이 너무 길어요"}, status_code=413)
    res, out = qc.analyze(data, p["text"], p["group"])
    d = user_dir(s["uid"])
    if out is not None:
        with open(os.path.join(d, "takes", f"{pid}.wav"), "wb") as f:
            f.write(out)
    with user_lock(s["uid"]):
        q = read_json(os.path.join(d, "qc.json"), {})
        q[pid] = {**res, "at": time.time()}
        write_json(os.path.join(d, "qc.json"), q)
    return res


@app.get("/api/takes")
def takes(req: Request):
    s = current(req)
    if not s:
        return need_login()
    q = read_json(os.path.join(user_dir(s["uid"]), "qc.json"), {})
    return {pid: {k: v.get(k) for k in ("verdict", "issues", "duration", "asr", "cer")} for pid, v in q.items()
            if pid in PROMPT_BY_ID}


@app.get("/api/take/{pid}")
def take_audio(pid: str, req: Request):
    s = current(req)
    if not s:
        return need_login()
    path = os.path.join(user_dir(s["uid"]), "takes", f"{pid}.wav")
    if pid not in PROMPT_BY_ID or not os.path.exists(path):
        return Response(status_code=404)
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.delete("/api/take/{pid}")
def take_delete(pid: str, req: Request):
    s = current(req)
    if not s:
        return need_login()
    d = user_dir(s["uid"])
    with user_lock(s["uid"]):
        q = read_json(os.path.join(d, "qc.json"), {})
        q.pop(pid, None)
        write_json(os.path.join(d, "qc.json"), q)
    try:
        os.remove(os.path.join(d, "takes", f"{pid}.wav"))
    except OSError:
        pass
    return {"ok": True}


def usable(uid):
    """학습에 쓸 수 있는 테이크(good/warn) 목록: [(prompt, qc)]"""
    d = user_dir(uid)
    q = read_json(os.path.join(d, "qc.json"), {})
    out = []
    for pid, r in q.items():
        p = PROMPT_BY_ID.get(pid)
        if p and r.get("verdict") in ("good", "warn") and os.path.exists(os.path.join(d, "takes", f"{pid}.wav")):
            out.append((p, r))
    return out


def progress_of(uid):
    q = read_json(os.path.join(user_dir(uid), "qc.json"), {})
    use = usable(uid)
    groups = {}
    for g in PROMPTS["groups"]:
        ps = [p for p in PROMPTS["prompts"] if p["group"] == g["id"]]
        groups[g["id"]] = {"total": len(ps), "recorded": sum(1 for p in ps if p["id"] in q),
                           "usable": sum(1 for p, _ in use if p["group"] == g["id"])}
    minutes = sum(r.get("duration", 0) for _, r in use) / 60
    reasons = []
    if groups["neutral"]["usable"] < MIN_NEUTRAL:
        reasons.append(f"평서문 {MIN_NEUTRAL}개 이상 (지금 {groups['neutral']['usable']}개)")
    if len(use) < MIN_TOTAL:
        reasons.append(f"전체 {MIN_TOTAL}개 이상 (지금 {len(use)}개)")
    if minutes < MIN_MINUTES:
        reasons.append(f"녹음 {MIN_MINUTES:.0f}분 이상 (지금 {minutes:.1f}분)")
    return {"groups": groups, "usable": len(use), "minutes": round(minutes, 1), "can_train": not reasons,
            "need": reasons}


@app.get("/api/progress")
def progress(req: Request):
    s = current(req)
    if not s:
        return need_login()
    return progress_of(s["uid"])


# ---------- 학습 ----------

@app.post("/api/train")
def train(req: Request):
    s = current(req)
    if not s:
        return need_login()
    pr = progress_of(s["uid"])
    if not pr["can_train"]:
        return JSONResponse({"error": "아직 녹음이 부족해요: " + ", ".join(pr["need"])}, status_code=400)
    d = user_dir(s["uid"])
    use = usable(s["uid"])
    version = 1 + sum(1 for j in trainer.jobs_for(s["uid"]) if j["status"] == "done")
    lst = os.path.join(d, f"train_v{version}.list")
    with open(lst, "w", encoding="utf-8") as f:
        for p, _ in use:
            f.write(f"{os.path.join(d, 'takes', p['id'] + '.wav')}|{s['uid']}|KO|{p['text']}\n")
    jid = trainer.submit(s["uid"], lst, len(use), pr["minutes"])
    if not jid:
        return JSONResponse({"error": "이미 학습이 대기 중이거나 진행 중이에요"}, status_code=409)
    return {"ok": True, "job": jid}


@app.get("/api/train")
def train_status(req: Request):
    s = current(req)
    if not s:
        return need_login()
    jobs = trainer.jobs_for(s["uid"])
    return {"jobs": [{k: j.get(k) for k in ("id", "version", "status", "stage", "progress", "items", "minutes",
                                             "created", "started", "finished", "error", "ahead")} for j in jobs],
            "gpu_busy": trainer.busy()}


# ---------- 합성 ----------

_rate = {}


def rate_ok(uid, per_min=20):
    now = time.time()
    q = [t for t in _rate.get(uid, []) if now - t < 60]
    if len(q) >= per_min:
        return False
    q.append(now)
    _rate[uid] = q
    return True


def refs_for(uid, style):
    """그 톤의 본인 녹음 중 레퍼런스로 좋은 것(3~10초, 깨끗한 순). 없으면 평서문."""
    d = user_dir(uid)
    cands = [(p, r) for p, r in usable(uid) if p["group"] != "reaction" and 3.0 <= r.get("duration", 0) <= 10.0]
    for st in (style, "neutral"):
        pick = sorted([c for c in cands if c[0]["style"] == st],
                      key=lambda c: (c[1]["verdict"] != "good", -c[1].get("snr", 0)))
        if pick:
            return [(os.path.join(d, "takes", p["id"] + ".wav"), p["text"]) for p, _ in pick[:2]], st
    return [], None


@app.get("/api/voices")
def voices(req: Request):
    s = current(req)
    if not s:
        return need_login()
    done = [j for j in trainer.jobs_for(s["uid"]) if j["status"] == "done"]
    cands = [p for p, r in usable(s["uid"]) if p["group"] != "reaction" and 3.0 <= r.get("duration", 0) <= 10.0]
    styles = sorted({p["style"] for p in cands}, key=lambda st: list(PROMPTS["styles"]).index(st))
    return {"versions": [{"version": j["version"], "finished": j.get("finished"), "items": j.get("items"),
                          "minutes": j.get("minutes")} for j in done],
            "styles": [{"id": st, "name": PROMPTS["styles"][st]} for st in styles]}


@app.post("/api/synth")
def synth_api(payload: dict, req: Request):
    s = current(req)
    if not s:
        return need_login()
    text = str(payload.get("text", "")).strip()[:300]
    if not text:
        return JSONResponse({"error": "읽을 문장을 넣어 주세요"}, status_code=400)
    if trainer.busy():
        return JSONResponse({"error": "지금 그래픽카드로 학습 중이라 잠시 후에 다시 해 주세요"}, status_code=503)
    if not rate_ok(s["uid"]):
        return JSONResponse({"error": "너무 빨라요. 잠시 후 다시 해 주세요"}, status_code=429)
    done = [j for j in trainer.jobs_for(s["uid"]) if j["status"] == "done"]
    if not done:
        return JSONResponse({"error": "아직 학습된 모델이 없어요"}, status_code=404)
    ver = int(payload.get("version") or done[-1]["version"])
    job = next((j for j in done if j["version"] == ver), done[-1])
    refs, used = refs_for(s["uid"], str(payload.get("style", "neutral")))
    if not refs:
        return JSONResponse({"error": "레퍼런스로 쓸 녹음(3~10초)이 없어요"}, status_code=400)
    import synth
    speed = min(1.5, max(0.7, float(payload.get("speed", 1.0))))
    t0 = time.time()
    try:
        data, mime = synth.synth(job["gpt"], job["sovits"], text, refs[0][0], refs[0][1],
                                 aux=[r[0] for r in refs[1:]], speed=speed)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"합성 실패: {str(e)[:200]}"}, status_code=500)
    return Response(data, media_type=mime, headers={"X-Style-Used": used, "X-Synth-Seconds": f"{time.time() - t0:.2f}",
                                                    "Cache-Control": "no-store"})


def _unload_synth():
    if "synth" in sys.modules:
        sys.modules["synth"].unload()


trainer.on_before_train.append(_unload_synth)


# ---------- 페이지 ----------

@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"), media_type="text/html; charset=utf-8",
                        headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    os.makedirs(USERS, exist_ok=True)
    trainer.start_worker()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
