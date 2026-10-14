"""GPT-SoVITS 추론 엔진 불러오기(v2ProPlus)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from settings import GSV_ROOT  # noqa: E402

sys.path.insert(0, GSV_ROOT)
sys.path.insert(0, os.path.join(GSV_ROOT, "GPT_SoVITS"))
os.chdir(GSV_ROOT)  # GPT-SoVITS는 상대 경로로 사전학습 모델을 찾는다
from TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PRETRAINED = "GPT_SoVITS/pretrained_models"


def load_tts(gpt, sovits):
    return TTS(TTS_Config({"custom": {
        "device": "cuda",
        "is_half": True,
        "version": "v2ProPlus",
        "t2s_weights_path": gpt,
        "vits_weights_path": sovits,
        "bert_base_path": f"{PRETRAINED}/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": f"{PRETRAINED}/chinese-hubert-base",
    }}))
