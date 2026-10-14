"""설정: 레포 최상위의 .env(없으면 환경변수)에서 읽는다. .env.example 참고."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")


def _load_env():
    try:
        with open(ENV_PATH, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except OSError:
        pass


_load_env()

# GPT-SoVITS 설치 위치(통합 패키지 폴더: runtime\python.exe 와 GPT_SoVITS\ 가 들어 있는 곳)
GSV_ROOT = os.environ.get("VOICELAB_GSV_ROOT", r"C:\GPT-SoVITS")
DATA = os.environ.get("VOICELAB_DATA", os.path.join(ROOT, "data"))
PORT = int(os.environ.get("VOICELAB_PORT", "8830"))
# 받아쓰기(대본 일치 검사)에 쓸 OpenAI 호환 /audio/transcriptions 주소와 키. 비우면 그 검사만 건너뛴다
ASR_URL = os.environ.get("VOICELAB_ASR_URL", "")
ASR_KEY = os.environ.get("VOICELAB_ASR_KEY", "")
# 같은 GPU를 쓰는 다른 서버가 있으면 학습 동안 내렸다가 다시 켠다.
# PAUSE_MATCH: 내릴 프로세스 명령줄에 들어 있는 글자(쉼표로 여러 개), RESUME_CMD: 다시 켤 명령
PAUSE_MATCH = [s for s in os.environ.get("VOICELAB_PAUSE_MATCH", "").split(",") if s.strip()]
RESUME_CMD = os.environ.get("VOICELAB_RESUME_CMD", "")
