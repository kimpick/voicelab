"""학습 작업: 사용자 녹음 → GPT-SoVITS 학습 목록 → 전처리 → SoVITS·GPT 미세조정 → 모델 등록.

- 한 번에 한 작업만(그래픽카드 8GB). 대기열은 data\\jobs.json에 남아 서버를 다시 켜도 이어진다.
- 같은 GPU를 쓰는 다른 서버가 있으면(.env의 VOICELAB_PAUSE_MATCH) 학습 동안 내리고, 끝나면 VOICELAB_RESUME_CMD로 다시 켠다.
- gsv\\train_gsv.py의 단계 함수를 그대로 쓴다(설정값만 바꿔서).
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gsv"))
import train_gsv  # noqa: E402
from settings import DATA, PAUSE_MATCH, RESUME_CMD  # noqa: E402

GSV_ROOT = train_gsv.GSV_ROOT
VERSION = train_gsv.VERSION
JOBS = os.path.join(DATA, "jobs.json")
# 10~20분 분량 한 사람 녹음 기준. 체크포인트는 마지막 것만 남긴다(사람마다 수백 MB씩 쌓이지 않게)
SOVITS = dict(batch_size=4, epochs=8, save_every_epoch=8, text_low_lr_rate=0.4)
GPT = dict(batch_size=4, epochs=15, save_every_epoch=15)
if os.environ.get("VOICELAB_EPOCHS"):  # 파이프라인 시험용으로 짧게: "2,3" → SoVITS 2, GPT 3 에폭
    _s, _g = (int(v) for v in os.environ["VOICELAB_EPOCHS"].split(","))
    SOVITS.update(epochs=_s, save_every_epoch=_s)
    GPT.update(epochs=_g, save_every_epoch=_g)
STAGES = [("prep", 0.0, 0.12), ("sovits", 0.12, 0.62), ("gpt", 0.62, 0.97)]
LOADER_WORKERS = int(os.environ.get("VOICELAB_LOADER_WORKERS", "2"))

_lock = threading.Lock()
_wake = threading.Event()
state = {"current": None}  # 지금 도는 작업 id
on_before_train = []  # 학습 직전에 부를 함수들(합성 모델 내리기 등)


def _load():
    try:
        with open(JOBS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _save(jobs):
    os.makedirs(DATA, exist_ok=True)
    tmp = JOBS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=1)
    os.replace(tmp, JOBS)


def _update(jid, **kw):
    with _lock:
        jobs = _load()
        for j in jobs:
            if j["id"] == jid:
                j.update(kw)
        _save(jobs)


def submit(uid, train_list, n_items, minutes):
    with _lock:
        jobs = _load()
        if any(j["uid"] == uid and j["status"] in ("queued", "running") for j in jobs):
            return None
        version = 1 + sum(1 for j in jobs if j["uid"] == uid and j["status"] == "done")
        jid = f"{uid}-{int(time.time())}"
        jobs.append({"id": jid, "uid": uid, "version": version, "status": "queued", "list": train_list,
                     "items": n_items, "minutes": round(minutes, 1), "created": time.time(),
                     "progress": 0.0, "stage": "대기 중", "log": ""})
        _save(jobs)
    _wake.set()
    return jid


def jobs_for(uid):
    jobs = _load()
    queue = [j for j in jobs if j["status"] == "queued"]
    out = []
    for j in jobs:
        if j["uid"] != uid:
            continue
        j = dict(j)
        if j["status"] == "queued":
            j["ahead"] = [q["id"] for q in queue].index(j["id"]) + (1 if state["current"] else 0)
        out.append(j)
    return out


def busy():
    return state["current"] is not None


# ---------- 같은 GPU를 쓰는 다른 서버 내리기/켜기(설정했을 때만) ----------

def shared_procs():
    if not PAUSE_MATCH:
        return []
    out = []
    for p in psutil.process_iter(["pid", "cmdline", "name"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        if p.pid != os.getpid() and any(m in cmd for m in PAUSE_MATCH):
            out.append(p)
    return out


def stop_shared():
    procs = shared_procs()
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=20)
    for p in procs:
        try:
            if p.is_running():
                p.kill()
        except psutil.Error:
            pass
    time.sleep(3)  # GPU 메모리가 풀릴 시간
    return bool(procs)


def start_shared():
    """내렸던 서버를 이 프로세스와 무관하게 다시 띄운다(WMI로 만들어 부모가 WmiPrvSE가 되게).
    보이스랩 서버의 자식으로 띄우면, 보이스랩을 끌 때 프로세스 트리째 같이 꺼진다."""
    if not RESUME_CMD:
        return
    cmd = RESUME_CMD.replace("'", "''")
    ps = ("Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
          f"@{{ CommandLine = '{cmd}' }} | Out-Null")
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], timeout=60,
                   creationflags=subprocess.CREATE_NO_WINDOW)


# ---------- 학습 실행 ----------

def _run_logged(jid, fn, log_path, base, span, epochs):
    """train_gsv의 단계 함수를 돌리되, 하위 프로세스 출력을 기록 파일로 받고 에폭 줄을 읽어 진행률을 갱신한다."""
    real_run = train_gsv.run

    def run(script, env_extra):
        env_extra = dict(env_extra)
        args = env_extra.pop("_args", [])
        env = os.environ.copy()
        env.update({k: str(v) for k, v in env_extra.items()})
        env["PATH"] = os.path.join(GSV_ROOT, "runtime") + os.pathsep + env["PATH"]
        env["PYTHONIOENCODING"] = "utf-8"
        # 데이터 로더 프로세스 수를 줄인다(윈도우는 로더마다 파이썬+torch 프로세스가 떠서 RAM 16GB PC에서 메모리가 넘쳤다)
        env["GSV_NUM_WORKERS"], env["GSV_VAL_WORKERS"] = str(LOADER_WORKERS), "1"
        if script.endswith("s1_train.py") and "--config_file" in args:
            import yaml
            cfg = args[args.index("--config_file") + 1]
            with open(cfg, encoding="utf-8") as f:
                y = yaml.load(f, Loader=yaml.FullLoader)
            y["data"]["num_workers"] = LOADER_WORKERS
            with open(cfg, "w", encoding="utf-8") as f:
                f.write(yaml.dump(y, default_flow_style=False))
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {script}\n")
            lf.flush()
            p = subprocess.Popen([train_gsv.PY, "-s", script] + args, cwd=GSV_ROOT, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", bufsize=1)
            last = 0.0
            for line in p.stdout:
                lf.write(line)
                m = re.search(r"Epoch[: ]+(\d+)", line)
                if m and epochs and time.time() - last > 2:
                    last = time.time()
                    e = min(int(m.group(1)), epochs)
                    _update(jid, progress=round(base + span * e / epochs, 3))
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"{os.path.basename(script)} 실패(코드 {p.returncode})")

    train_gsv.run = run
    try:
        fn()
    finally:
        train_gsv.run = real_run


def _find_weight(folder, exp, ext):
    d = os.path.join(GSV_ROOT, folder)
    cands = [os.path.join(d, f) for f in os.listdir(d) if f.startswith(exp) and f.endswith(ext)]
    return max(cands, key=os.path.getmtime) if cands else None


def run_job(job):
    jid, uid = job["id"], job["uid"]
    exp = f"vl_{uid}_v{job['version']}"
    user_dir = os.path.join(DATA, "users", uid)
    log_path = os.path.join(user_dir, f"train_v{job['version']}.log")
    _update(jid, status="running", started=time.time(), stage="그래픽카드 준비", progress=0.01, exp=exp)
    for fn in on_before_train:
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass
    shared_was_on = stop_shared()
    try:
        train_gsv.EXP_NAME = exp
        train_gsv.TRAIN_LISTS = [job["list"]]
        train_gsv.SOVITS, train_gsv.GPT = SOVITS, GPT
        train_gsv.OPT_DIR = f"logs/{exp}"
        train_gsv.INP_TEXT = os.path.join(user_dir, f"train_v{job['version']}_all.list")
        names = {"prep": "데이터 전처리", "sovits": "음색 학습(SoVITS)", "gpt": "말투 학습(GPT)"}
        for stage, base, top in STAGES:
            _update(jid, stage=names[stage], progress=base)
            epochs = SOVITS["epochs"] if stage == "sovits" else GPT["epochs"] if stage == "gpt" else 0
            _run_logged(jid, getattr(train_gsv, stage), log_path, base, top - base, epochs)
        sovits_w = _find_weight(f"SoVITS_weights_{VERSION}", exp + "_", ".pth")
        gpt_w = _find_weight(f"GPT_weights_{VERSION}", exp + "-", ".ckpt")
        if not (sovits_w and gpt_w):
            raise RuntimeError("학습은 끝났는데 모델 파일을 찾지 못했어요")
        _update(jid, status="done", progress=1.0, stage="완료", finished=time.time(),
                sovits=sovits_w, gpt=gpt_w)
    except Exception as e:  # noqa: BLE001
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n" + traceback.format_exc())
        _update(jid, status="error", stage="실패", error=str(e)[:300], finished=time.time())
    finally:
        if shared_was_on:
            start_shared()


def worker():
    # 서버가 꺼졌다 켜지면서 '실행 중'으로 남은 작업은 다시 대기열로
    with _lock:
        jobs = _load()
        for j in jobs:
            if j["status"] == "running":
                j.update(status="queued", stage="대기 중(서버 재시작)", progress=0.0)
        _save(jobs)
    while True:
        nxt = next((j for j in _load() if j["status"] == "queued"), None)
        if not nxt:
            _wake.wait(30)
            _wake.clear()
            continue
        state["current"] = nxt["id"]
        try:
            run_job(nxt)
        finally:
            state["current"] = None


def start_worker():
    t = threading.Thread(target=worker, daemon=True, name="voicelab-trainer")
    t.start()
    return t
