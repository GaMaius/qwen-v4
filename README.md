# 아주 소중한 딥러닝 챌린지 2026 — 수학 문제 풀이 모델

`Qwen/Qwen2.5-3B-Instruct` 를 QLoRA 로 파인튜닝하고, vLLM 다수결 투표로
정수 답을 추론한다. 학습·추론 전 과정을 이 저장소로 재현할 수 있다.

---

## 1. 모델 가중치

| | |
|---|---|
| 베이스 | `Qwen/Qwen2.5-3B-Instruct` |
| 파인튜닝 결과 (merged, fp16) | **https://www.kaggle.com/datasets/gamaius/qwen-v4** |

> 다른 모델의 가중치를 로드하거나 병합하지 않았다. LoRA 어댑터를 베이스에
> 병합한 것이 전부다 (규칙 4.1 / 4.3).

---

## 2. 실행 환경

추론은 **인터넷 차단 상태**에서 동작한다 (규칙 6a). 모든 휠을 사전에 받아
로컬 디렉터리에서 설치한다.

| | |
|---|---|
| Python | 3.12.13 |
| GPU | Tesla T4 x 2 (compute capability 7.5) |
| torch | 2.13.0+cu130 |
| vLLM | 0.27.1 |
| transformers | 5.15.1 |
| 학습 | unsloth + peft (QLoRA 4bit), Tesla T4 x 1 |

T4 는 FlashAttention-2 를 지원하지 않아 vLLM 이 `TRITON_ATTN` 백엔드로
자동 전환하며, FlashInfer 샘플러도 비활성화된다. 정상 동작이다.

필요한 휠 195개를 미리 받아 **공개 데이터셋으로 올려두었다.** 인터넷 없이
그대로 설치된다.

- 휠 데이터셋: <https://www.kaggle.com/datasets/gamaius/qwen-probe> (`vllm_wheels/`)

```bash
# 캐글 노트북(Internet: Off)에서 그대로 실행된다
WHEELS=$(dirname $(find /kaggle/input -name '*.whl' | head -1))
pip install --no-index --find-links=$WHEELS vllm torchvision
pip uninstall -y torchaudio   # torch(cu130)와 CUDA 버전이 어긋나 vLLM 로드를 막는다
```

`--no-index` 가 없으면 인터넷 차단 상태에서 PyPI 를 조회하려다 실패한다.
설치에 약 260초가 걸린다.

> 휠은 동일한 캐글 이미지(Python 3.12.13)에서 `pip download vllm torchvision`(당시 vLLM 0.27.1)
> 으로 만들었다. 인터프리터 버전이 다르면 `cp312` 휠이 맞지 않으므로,
> 환경을 바꿔 재현할 때는 그 환경에서 다시 받아야 한다.

---

## 3. 학습 데이터

### 3.1 주최측 제공 데이터
- 대회 train 데이터셋을 사용했다. 라벨 정제는 `01_clean_data.py` 에서
  수행하며, 자기일관성 투표로 따로 검출한 오답 라벨 목록을
  `bad_label_ids.csv` 로 남겼다.
- test 데이터는 학습에 **사용하지 않았다** (규칙 5.1b).

### 3.2 외부 공개 데이터셋 (규칙 5.2c 명시)

| 데이터셋 | 라이선스 | 사용량 |
|---|---|---|
| [`nvidia/OpenMathInstruct-2`](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | CC-BY-4.0 | 25,000 |
| [`AI-MO/NuminaMath-CoT`](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Apache-2.0 | 7,000 |
| [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | MIT | 2,350 |

모두 무료로 동등하게 접근 가능한 공개 데이터셋이다.

### 3.3 상용 API 사용 내역 (규칙 5.3a)
대회 **train** 문항의 풀이(CoT) 생성에 상용 LLM API 를 사용했다
(`competition_gemini_cot`, 5,919건). test 문항에는 사용하지 않았다.

---

## 4. 재현 절차

### 4.1 데이터 준비
```bash
# 라벨 정제 · 포맷 통일
#   → deep_chal_math_clean_v4.csv, local_val_500.csv(자체 홀드아웃 500문항)
python 01_clean_data.py

# 외부 공개 데이터셋 수집 · 병합
#   → deep_chal_math_clean_v5.csv (40,269행) — 학습 입력
python 04_add_external_data.py --base deep_chal_math_clean_v4.csv
```

`local_val_500.csv` 는 train 에서 떼어낸 홀드아웃이며 학습에 넣지 않는다.
설정 선택(온도·토큰 길이·앙상블 여부)은 전부 이 500문항에서 결정했고,
리더보드 제출로 튜닝하지 않았다.

> `10_find_bad_labels.py` 는 자기일관성 생성 결과(`rft_out/`)를 입력으로
> 오답 라벨을 찾는 진단 도구다. 그 생성 단계는 최종 모델 학습 경로에
> 포함되지 않으므로, 산출물인 `bad_label_ids.csv` 만 저장소에 넣었다.

### 4.2 학습
```bash
DATA=deep_chal_math_clean_v5.csv RUN=qwen_v4 LR=2e-4 EPOCHS=1 python 03_train_sft.py
```

| 하이퍼파라미터 | 값 |
|---|---|
| 방식 | QLoRA (4bit), unsloth |
| LoRA r / alpha / dropout | 32 / 64 / 0 |
| target modules | q,k,v,o,gate,up,down_proj |
| learning rate | 2e-4 |
| epochs | 1 |
| max_seq_length | 1536 |
| effective batch | 16 |
| gradient checkpointing | unsloth |
| seed | 42 |

학습 후 LoRA 를 베이스에 병합해 fp16 으로 저장한다 (`*_merged`).

### 4.3 추론 — 제출 파일 재현

제출 파일은 **캐글 노트북에서** 만들었다. 아래 구성을 그대로 재현하면 된다.

#### 노트북 설정

| 항목 | 값 |
|---|---|
| Accelerator | **GPU T4 x 2** (P100 은 compute capability 6.0 이라 vLLM 이 뜨지 않는다) |
| Internet | **Off** (규칙 6a) |
| Environment | 기본 이미지 (Python 3.12.13) |

#### Input 에 붙일 것 4개

| # | 대상 | 경로 |
|---|---|---|
| 1 | 파인튜닝 모델 | Datasets → `gamaius/qwen-v4` |
| 2 | 오프라인 휠 | Datasets → `gamaius/qwen-probe` (내부에 `vllm_wheels/`) |
| 3 | 이 저장소 | 아래 참고 |
| 4 | 주최측 `test_submission.csv` | 대회 Data 탭 |

3번은 저장소를 내려받아 캐글 데이터셋으로 올리면 된다.
`kaggle_inference.py` 는 `02_infer_vllm.py` 를 **자기 파일 옆 → `/kaggle/working` →
`/kaggle/input`** 순으로 찾으므로, 데이터셋으로 붙이든 노트북에서 직접 풀어놓든
동작한다.

```bash
git clone https://github.com/GaMaius/qwen-v4
cd qwen-v4 && kaggle datasets init -p . && kaggle datasets create -p .
```

#### 실행

제출에 실제로 사용한 노트북이 `kaggle_notebook_submission.ipynb` 다.
캐글에 그대로 올려 위 Input 4개를 붙이면 된다.
새로 만들려면 셀 두 개면 충분하다.

```python
# 셀 1 — 오프라인 설치 (약 260초)
import glob, os, sys
W = os.path.dirname(sorted(glob.glob("/kaggle/input/**/*.whl", recursive=True))[0])
!{sys.executable} -m pip install --no-index --find-links={W} vllm torchvision
!{sys.executable} -m pip uninstall -y torchaudio
```

```python
# 셀 2 — 추론 (약 4시간). /kaggle/working/submission.csv 가 나온다
%run /kaggle/input/<저장소-데이터셋>/kaggle_inference.py
```

`kaggle_inference.py` 는 실행 전에 입력 파일·모델 폴더·가중치 크기·vLLM 설치를
모두 `assert` 로 확인한다. **특히 모델은 `qwen_v4_merged` 로 이름을 고정한다** —
노트북에 베이스 Qwen2.5-3B-Instruct 가 함께 붙어 있으면 그쪽이 먼저 잡혀,
파인튜닝되지 않은 모델로 4시간을 돌리게 된다.

아래는 그 스크립트가 실제로 실행하는 명령이다.


**GPU 2장에 문항을 반씩 나눠 독립 프로세스로 돌린다.** 샤딩은
`df.iloc[shard::num_shards]` 이므로, 단일 GPU 로 돌리면 배치 구성이 달라져
결과가 재현되지 않는다.

```bash
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i python -u 02_infer_vllm.py \
    --model <merged_model_dir> \
    --input test_submission.csv \
    --output sub_$i.csv \
    --shard $i --num-shards 2 \
    --n 32 --temperature 0.5 --top-p 0.95 \
    --max-tokens 2048 --max-model-len 3072 \
    --chunk 50 &
done; wait

# 두 샤드를 합쳐 원본 순서로 정렬 (kaggle_inference.py 가 하는 일과 동일)
python - <<'EOF'
import pandas as pd
ref = pd.read_csv("test_submission.csv")
sub = pd.concat([pd.read_csv(f"sub_{i}.csv") for i in range(2)])         .drop_duplicates("id", keep="last")
sub = ref[["id"]].merge(sub[["id", "answer"]], on="id")
assert len(sub) == len(ref) and sub["answer"].notna().all()
sub["answer"] = sub["answer"].astype("int64")
sub.to_csv("submission.csv", index=False)
EOF

# 형식 검사 (행 수 · id 집합 · 정수 여부)
python 05_check_submission.py --sub submission.csv --ref test_submission.csv
```

> 주최측이 대문자 `ID` 컬럼을 요구하면
> `--id-col ID` 를 붙인다. `submission_fixed.csv` 로 이름을 바꿔 저장한다.

| 샘플링 파라미터 | 값 |
|---|---|
| n (self-consistency) | 32 |
| temperature | 0.5 |
| top_p | 0.95 |
| repetition_penalty | 1.0 |
| max_tokens | 2048 |
| max_model_len | 3072 |
| dtype | float16 |
| enforce_eager | True |
| seed | 1234 |

소요 시간: 2,000문항 기준 T4 x 2 에서 약 4시간.

---

## 5. 답 추출과 투표

모델 출력에서 정수를 뽑는 후처리는 직접 구현했다 (규칙 7.2d).

1. `\boxed{}` 안의 값 — 등급 2
2. 꼬리 문장의 "answer is N" — 등급 1
3. 마지막 줄의 숫자 — 등급 0

**투표는 가장 높은 등급의 샘플들끼리만 한다.** `\boxed{}` 를 낸 샘플이
하나라도 있으면 폴백으로 건진 답은 표에서 제외된다. 잘린 출력의 마지막
숫자가 표를 오염시키는 것을 막기 위함이다.

`|answer| > 10^15` 는 환각으로 보고 버린다.

---

## 6. 주요 실험 결과

Public 리더보드(831문항) 실측. 자세한 내용은 [METHODOLOGY.md](METHODOLOGY.md).

| 설정 | Public |
|---|---|
| temp 0.7, max_tokens 2048, n=32 | 0.75812 |
| temp 0.7, max_tokens 4096, n=32 | 0.74368 |
| temp 0.7 + 다른 모델 앙상블, n=64 | 0.75090 |
| temp 0.7 + 프롬프트 다양성, n=64 | 0.75451 |
| **temp 0.5, max_tokens 2048, n=32** | **0.77617** |

> 위 앙상블 실험의 두 모델은 **모두 `Qwen/Qwen2.5-3B-Instruct` 에서
> 파인튜닝한 것**이며, 외부 모델을 호출하지 않았다 (규칙 4.3).
> 그리고 이득이 없어 **최종 제출에는 사용하지 않았다.** 제출 구성은
> 단일 모델 + self-consistency 다수결이다.

배운 것:

- **앙상블은 도움이 되지 않았다.** 홀드아웃에서 관측한 이득은 대부분
  *표본 수 차이*였다. 표본 수를 맞춘 대조군(같은 모델·시드만 변경)을
  세우자 이득이 사라졌다.
- **출력 잘림(12.8%)의 원인은 토큰 예산이 아니라 온도였다.**
  `max_tokens` 를 2048 → 4096 으로 늘려도 10.9% 였지만,
  temperature 를 0.7 → 0.5 로 내리자 **2.1%** 로 떨어졌다.
  높은 온도에서 모델이 반복 루프에 빠져 있었던 것이다.
- 학습 데이터의 풀이 길이 중앙값은 361토큰, 90%가 736토큰 이하다.
  모델은 긴 풀이를 생성하도록 학습된 적이 없다.

---

## 7. 파일 구성

| 파일 | 역할 |
|---|---|
| **`kaggle_notebook_submission.ipynb`** | **제출에 실제로 사용한 캐글 노트북 원본** |
| `kaggle_inference.py` | 위 노트북을 한 파일로 합친 것 — 설치·추론·검증 전 과정 |
| `02_infer_vllm.py` | 추론 · 다수결 투표 · 답 추출 |
| `05_check_submission.py` | 제출 파일 검증 |
| `01_clean_data.py` | 라벨 정제, 포맷 통일 |
| `03_train_sft.py` | QLoRA SFT |
| `04_add_external_data.py` | 외부 공개 데이터셋 수집 |
| `10_find_bad_labels.py` | 오답 라벨 검출 |
| `bad_label_ids.csv` | 검출된 오답 라벨의 id 목록 |
| `METHODOLOGY.md` | 실험 기록 전문 |

> 주최측이 제공한 데이터(train / test CSV)와 그로부터 파생한 홀드아웃
> 500문항은 재배포하지 않는다. `01_clean_data.py` 와 `10_find_bad_labels.py`
> 로 동일하게 생성된다.
