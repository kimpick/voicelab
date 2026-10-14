# VoiceLab — 내 목소리로 TTS 만들기

브라우저에서 분류된 대본(평서문·감정·연기톤·리액션)을 15~20분 녹음하면, 그 목소리로 글을 읽는 개인 TTS 모델을 만들어 주는 사내용 웹 서비스입니다.
학습·합성은 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v2ProPlus 미세조정을 씁니다.

## 흐름

1. **로그인** — 구글 계정. `allowed_domains`(회사 도메인) 또는 `allowed_emails`(개별 허용)만 들어올 수 있습니다.
2. **동의 녹음** — 본인 목소리로 동의 문장을 읽습니다. 녹음은 화면에서만 할 수 있고 파일 업로드는 없습니다(남의 목소리 도용 방지).
3. **녹음 스튜디오** — `app/prompts.json` 188문장
   - 평서문 50
   - 감정 52: 기쁨·슬픔·화남·놀람·불안·다정
   - 연기톤 45: 내레이션·아나운서·속삭임·귀엽게·장난·지침·외치기
   - 리액션 41: 웃음·감탄·헉·맞장구·한숨·생각
4. **자동 품질검사** (`app/qc.py`) — 테이크마다 앞뒤 무음 자르기, 찢어짐, 음량, 잡음(SNR), 말 속도, 받아쓰기로 대본 일치(CER) → good / warn / bad(학습 제외)
5. **학습** (`app/trainer.py`) — 평서문 30개·전체 80개·6분 이상이면 시작. 한 번에 한 명씩 대기열로 돌고, 다시 학습하면 새 버전이 쌓입니다.
6. **들어보기** (`app/synth.py`) — 톤을 고르면 그 톤으로 녹음한 본인 목소리(3~10초, 깨끗한 순)를 레퍼런스로 읽습니다. 속도·버전 선택.

## 설치

1. GPT-SoVITS 통합 패키지(Windows, `runtime\python.exe` 포함)를 받고 v2ProPlus 사전학습 모델을 둡니다.
2. `patches/gpt-sovits-loader-workers.patch` 의 두 줄을 GPT-SoVITS에 적용합니다(RAM 16GB PC에서 학습이 메모리 부족으로 죽는 문제).
3. `.env.example` → `.env`, `app/config.example.json` → `app/config.json` 으로 복사하고 값을 채웁니다.
4. `start_voicelab.bat` 실행 → http://127.0.0.1:8830

## 구조

```
app/        server.py(웹·API) qc.py(품질검사) trainer.py(학습 대기열) synth.py(합성) settings.py prompts.json static/index.html
gsv/        train_gsv.py(GPT-SoVITS 학습 단계) tts_engine.py(추론 엔진) fast_t2s.py·tts_speedups.py(합성 가속: GPT 토큰 생성 CUDA Graph, SoVITS 길이 구간화)
patches/    GPT-SoVITS 로더 프로세스 수 패치
data/       (git 제외) 사용자별 녹음 takes/, qc.json, meta.json, 학습 기록, 대기열 jobs.json
```

## 참고

- 그래픽카드 8GB 한 장 기준입니다. 같은 GPU를 쓰는 다른 서버가 있으면 `.env`의 `VOICELAB_PAUSE_MATCH` / `VOICELAB_RESUME_CMD` 로 학습 동안 내렸다가 다시 켭니다.
- 학습 시간: 녹음 8분 기준 약 20분(RTX 3060 Ti). 합성: 첫 문장 모델 로딩 약 20초, 이후 문장당 1~2초.
- 녹음과 모델은 개인의 목소리입니다. 서버 밖으로 내보내지 않도록 `data/`와 모델 폴더 접근을 관리하세요.
