"""GPT-SoVITS 학습 단계(voicelab trainer가 설정값을 채워 부른다): WebUI의 '1-GPT-SoVITS-TTS' 탭과 같은 순서와 설정으로 스크립트를 실행한다.

단계
- prep: 학습 목록 합치기 → 1a 텍스트/BERT → 1b HuBERT·32k 음성 → 1b SV 임베딩(v2Pro 계열) → 1c 시맨틱 토큰
- sovits: SoVITS(s2) 학습
- gpt: GPT(s1) 학습

"""
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from settings import GSV_ROOT  # noqa: E402
VERSION = "v2ProPlus"
PRETRAINED = "GPT_SoVITS/pretrained_models"
PY = os.path.join(GSV_ROOT, "runtime", "python.exe")



def run(script, env_extra):
    env_extra = dict(env_extra)
    args = env_extra.pop("_args", [])
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items()})
    env["PATH"] = os.path.join(GSV_ROOT, "runtime") + os.pathsep + env["PATH"]
    print(f"\n=== {script}", flush=True)
    subprocess.run([PY, "-s", script] + args, cwd=GSV_ROOT, env=env, check=True)


def prep():
    lines = []
    for p in TRAIN_LISTS:
        with open(p, encoding="utf-8") as f:
            lines += [ln for ln in f.read().splitlines() if ln.strip()]
    with open(INP_TEXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"train list: {len(lines)} lines -> {INP_TEXT}")

    common = {
        "inp_text": INP_TEXT,
        "inp_wav_dir": "",
        "exp_name": EXP_NAME,
        "opt_dir": OPT_DIR,
        "i_part": 0,
        "all_parts": 1,
        "_CUDA_VISIBLE_DEVICES": 0,
        "is_half": True,
        "version": VERSION,
    }
    opt_abs = os.path.join(GSV_ROOT, OPT_DIR)

    run("GPT_SoVITS/prepare_datasets/1-get-text.py",
        {**common, "bert_pretrained_dir": f"{PRETRAINED}/chinese-roberta-wwm-ext-large"})
    os.replace(os.path.join(opt_abs, "2-name2text-0.txt"), os.path.join(opt_abs, "2-name2text.txt"))

    hubert_env = {**common, "cnhubert_base_dir": f"{PRETRAINED}/chinese-hubert-base",
                  "sv_path": f"{PRETRAINED}/sv/pretrained_eres2netv2w24s4ep4.ckpt"}
    run("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", hubert_env)
    run("GPT_SoVITS/prepare_datasets/2-get-sv.py", hubert_env)

    run("GPT_SoVITS/prepare_datasets/3-get-semantic.py",
        {**common, "pretrained_s2G": f"{PRETRAINED}/v2Pro/s2G{VERSION}.pth",
         "s2config_path": f"GPT_SoVITS/configs/s2{VERSION}.json"})
    part = os.path.join(opt_abs, "6-name2semantic-0.tsv")
    with open(part, encoding="utf-8") as f:
        body = f.read().strip("\n")
    with open(os.path.join(opt_abs, "6-name2semantic.tsv"), "w", encoding="utf-8") as f:
        f.write("item_name\tsemantic_audio\n" + body + "\n")
    os.remove(part)


def sovits():
    with open(os.path.join(GSV_ROOT, f"GPT_SoVITS/configs/s2{VERSION}.json")) as f:
        data = json.load(f)
    s2_dir = f"logs/{EXP_NAME}"
    os.makedirs(os.path.join(GSV_ROOT, s2_dir, f"logs_s2_{VERSION}"), exist_ok=True)
    data["train"].update({
        "batch_size": SOVITS["batch_size"],
        "epochs": SOVITS["epochs"],
        "text_low_lr_rate": SOVITS["text_low_lr_rate"],
        "pretrained_s2G": f"{PRETRAINED}/v2Pro/s2G{VERSION}.pth",
        "pretrained_s2D": f"{PRETRAINED}/v2Pro/s2D{VERSION}.pth",
        "if_save_latest": True,
        "if_save_every_weights": True,
        "save_every_epoch": SOVITS["save_every_epoch"],
        "gpu_numbers": "0",
        "grad_ckpt": False,
        "lora_rank": "32",
    })
    data["model"]["version"] = VERSION
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
    data["save_weight_dir"] = f"SoVITS_weights_{VERSION}"
    data["name"] = EXP_NAME
    data["version"] = VERSION
    cfg = os.path.join(GSV_ROOT, "TEMP", "tmp_s2.json")
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    with open(cfg, "w") as f:
        json.dump(data, f)
    run("GPT_SoVITS/s2_train.py", {"version": VERSION, "_args": ["--config", cfg]})


def gpt():
    with open(os.path.join(GSV_ROOT, "GPT_SoVITS/configs/s1longer-v2.yaml")) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    s1_dir = f"logs/{EXP_NAME}"
    os.makedirs(os.path.join(GSV_ROOT, s1_dir, "logs_s1"), exist_ok=True)
    data["train"].update({
        "batch_size": GPT["batch_size"],
        "epochs": GPT["epochs"],
        "save_every_n_epoch": GPT["save_every_epoch"],
        "if_save_every_weights": True,
        "if_save_latest": True,
        "if_dpo": False,
        "half_weights_save_dir": f"GPT_weights_{VERSION}",
        "exp_name": EXP_NAME,
    })
    data["pretrained_s1"] = f"{PRETRAINED}/s1v3.ckpt"
    data["train_semantic_path"] = f"{s1_dir}/6-name2semantic.tsv"
    data["train_phoneme_path"] = f"{s1_dir}/2-name2text.txt"
    data["output_dir"] = f"{s1_dir}/logs_s1_{VERSION}"
    cfg = os.path.join(GSV_ROOT, "TEMP", "tmp_s1.yaml")
    with open(cfg, "w") as f:
        f.write(yaml.dump(data, default_flow_style=False))
    run("GPT_SoVITS/s1_train.py",
        {"version": VERSION, "_CUDA_VISIBLE_DEVICES": 0, "hz": "25hz", "_args": ["--config_file", cfg]})
