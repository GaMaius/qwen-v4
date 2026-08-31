"""
03_train_sft.py  —  Colab(Unsloth) SFT 재학습. qwen_v3.ipynb의 CELL 3 대체.

기존 학습 대비 변경점:
  1) 학습 데이터: deep_chal_math_clean_v4.csv (01_clean_data.py 산출물)
     → 기존엔 formatted_text 전체 19,074개를 그대로 썼는데, 그중 30%가 \boxed{} 없는
       Gemini 사고로그였고 35%가 정답을 10회 이상 재진술하는 장황한 텍스트였다.
       모델이 배운 건 "수학"이 아니라 "혼잣말"이었다.
  2) learning_rate 2e-5 → 1e-5.
     3B Instruct 모델에 2e-5는 높다. Instruct 정렬이 깨지면서 지시 이행(=포맷 준수)
     능력이 먼저 망가진다.
  3) epochs 1 → 3. 정제 후 8.2k로 줄었으므로 에폭을 늘려 보상.
  4) LoRA r 16 → 32 (alpha 64). 수학 추론은 표현력이 더 필요하다.
  5) DPO 단계 삭제.
     164쌍은 통계적으로 노이즈이고, 애초에 망가진 SFT 위에 얹은 것이라 효과가 없다.
     RL을 쓸 거면 DPO가 아니라 정답 검증이 가능한 GRPO를 SFT 이후에 붙이는 게 맞다.

Colab A100/L4 기준 3 epoch 약 1.5~3시간.
"""

# ══════════════════════════════════════════════════════════
# [CELL 1] 설치 (최초 1회)
# ══════════════════════════════════════════════════════════
# !pip install -q unsloth

# ══════════════════════════════════════════════════════════
# [CELL 2] 학습
# ══════════════════════════════════════════════════════════
# ⚠️ GPU 1장만 보이게 고정한다. unsloth 를 import 하기 전에 해야 효과가 있다.
#    (CUDA 초기화 후에는 CUDA_VISIBLE_DEVICES 변경이 먹지 않는다)
#
#    unsloth 무료판은 멀티 GPU 학습을 지원하지 않는데, GPU가 2장 "보이기만 해도"
#    레이어를 두 장에 나눠 올린 뒤 임베딩 조회에서 터진다:
#      RuntimeError: Expected all tensors to be on the same device,
#      but got index is on cuda:0, different from other tensors on cuda:1
#    어차피 1장만 쓰므로 속도 손해는 없다. 2장을 쓰려면 --shard 방식의 추론에서만.
import os as _os                                           # noqa: E402
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ⚠️ unsloth 를 반드시 trl/transformers/peft 보다 "먼저" import 해야 한다.
#    나중에 하면 패치가 안 걸려서 느려지고 메모리를 더 쓴다 (unsloth가 경고를 띄운다).
#    아래 import 순서를 재정렬하지 말 것.
from unsloth import FastLanguageModel                      # noqa: I001  (반드시 최상단)
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import glob            # noqa: E402
import inspect         # noqa: E402
import os              # noqa: E402
import shutil          # noqa: E402

import pandas as pd    # noqa: E402
import torch           # noqa: E402
import transformers    # noqa: E402
import trl             # noqa: E402
from datasets import Dataset          # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

print(f"trl {trl.__version__} / transformers {transformers.__version__} / torch {torch.__version__}")

# vLLM을 이 노트북에 설치하면 torch가 CUDA 13 빌드로 교체되면서 환경이 깨진다
# (libnvrtc.so.13 없음 → torchcodec 로드 실패 → unsloth import 크래시).
# 학습에는 vLLM이 필요 없다. 추론(02_infer_vllm.py)은 별도 노트북에서 돌릴 것.
if not torch.cuda.is_available():
    raise SystemExit(
        "GPU가 잡히지 않습니다.\n"
        "  · 노트북 설정에서 Accelerator가 GPU(T4)인지 확인하세요.\n"
        "  · 이 노트북에 vLLM을 설치했다면 torch가 깨진 것입니다. "
        "새 노트북에서 vLLM 없이 다시 시작하세요."
    )
print(f"GPU: {torch.cuda.get_device_name(0)}")


def supported(cls):
    """클래스가 실제로 받는 인자 이름 집합 (라이브러리 버전마다 달라서 필요)"""
    s = set(inspect.signature(cls.__init__).parameters)
    return s | set(getattr(cls, "__dataclass_fields__", {}))


def pick(names, cls, value):
    """names 중 cls가 지원하는 첫 인자로 {이름: 값} 반환. 없으면 빈 dict"""
    have = supported(cls)
    for n in names:
        if n in have:
            return {n: value}
    print(f"경고: {cls.__name__} 가 {names} 중 아무것도 지원하지 않습니다 — 생략합니다")
    return {}

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"   # 규칙상 고정. 변경 금지.

# Qwen 토크나이저 실측: clean_v4 전체 텍스트 p99 = 1,268토큰, 최대 ~1,400.
# 1536이면 100% 온전히 들어간다. 2048은 순수 낭비(속도만 손해).
# 더 줄이면 안 되는 이유: 잘린 샘플은 \boxed{} 없이 끝나서
# 지금 고치려는 "정답 포맷 안 내놓는 병"을 그대로 다시 가르친다.
MAX_SEQ_LENGTH = 1536

# 40k × 1epoch. 20k × 2epoch과 비용은 같은데 데이터 다양성이 두 배라 SFT에서 더 유리하다.
# clean_v4(8.2k)만 쓸 거면 3으로 올릴 것.
EPOCHS = float(os.environ.get("EPOCHS", 1))

# ── 학습률 ────────────────────────────────────────────────────────────
# 1e-5 → 2e-4. 이 값이 지금까지 성능을 묶어두고 있었을 가능성이 높다.
#
#   · 이 스크립트는 load_in_4bit=True, 즉 QLoRA다. QLoRA 표준 학습률은 2e-4이고
#     1e-5는 그보다 20배 낮다. 4bit로 얼어붙은 베이스에 어댑터만 겨우 흔드는 수준이다.
#   · 40k × 1epoch × effective batch 16 = 약 2,500 step. 스텝 수도 넉넉하지 않아서
#     낮은 학습률을 스텝 수로 벌충하지도 못한다.
#   · 원래 2e-5에서 1e-5로 낮춘 이유는 "Instruct 정렬 보호"였는데, 이전 파인튜닝이
#     망가진 원인은 학습률이 아니라 데이터였다(30%가 \boxed{} 없이 끝나는 Gemini 사고로그).
#     원인이 아닌 변수를 과보정한 것이다.
#   · 정황 증거: 순정 0.538 → 학습 후 0.728의 상승분 대부분이 "\boxed{}를 내놓는 포맷 학습"
#     으로 설명된다. 정작 추론 능력은 거의 안 올랐다는 뜻이고, 이건 저학습(undertrain) 징후다.
#
# 본 학습 전에 반드시 10k 부분집합으로 짧게 검증할 것 (SUBSET 환경변수).
LEARNING_RATE = float(os.environ.get("LR", 2e-4))

# 학습률 검증용. SUBSET=10000 으로 주면 10k만 뽑아 짧게 돌린다.
# 한 번뿐인 10시간짜리 본 학습을 눈감고 던지지 않기 위한 안전장치.
SUBSET = int(os.environ.get("SUBSET", 0))

# 캐글/코랩 자동 감지.
# 캐글 권장: 12시간 백그라운드 실행(Save & Run All)이라 코랩 free의 잦은 끊김이 없고,
#           학습 산출물을 그대로 Dataset으로 내보내 추론 노트북에 바로 붙일 수 있다.
#           (구글드라이브 다운로드 → 캐글 업로드 왕복이 사라짐)
ON_KAGGLE = os.path.exists("/kaggle/working")

# 학습 데이터와 산출물 이름을 환경변수로 바꿀 수 있게 한다.
# 캐글 노트북에서 스크립트를 편집하지 않고 검증용/본학습용을 오갈 수 있어야 한다.
#   검증: DATA=probe_10k.csv    RUN=probe   LR=2e-4 SUBSET=0
#   본  : DATA=train_mix_v6.csv RUN=qwen_v5 LR=<검증에서 확정한 값>
DATA_ENV = os.environ.get("DATA")           # 파일명 또는 전체 경로
RUN = os.environ.get("RUN", "qwen_v4")      # 산출물 폴더 이름 (검증본이 본학습본을 덮지 않게)

if ON_KAGGLE:
    OUTPUT_DIR = f"/kaggle/working/{RUN}"
    want = DATA_ENV or "deep_chal_math_clean_v5.csv"
    DATA = want if os.path.isabs(want) else f"/kaggle/working/{want}"
    if not os.path.isfile(DATA):
        # /kaggle/input 은 데이터셋마다 폴더가 갈리므로 이름으로 전체를 뒤진다.
        hits = glob.glob(f"/kaggle/**/{os.path.basename(want)}", recursive=True)
        if not hits:
            raise FileNotFoundError(
                f"{want} 가 없습니다. Input에 추가했는지 확인하세요.\n"
                "  주의: /kaggle/working 은 세션이 끝나면 지워집니다. 생성 스크립트와 03을 "
                "같은 노트북에서 연달아 실행하거나, 결과를 Dataset으로 저장한 뒤 Input에 추가하세요."
            )
        DATA = hits[0]
        print(f"자동 탐색: {DATA}")
    MERGE_DIR = f"/kaggle/working/{RUN}_merged"    # 노트북 Output → Dataset으로 저장됨
else:
    from google.colab import drive
    drive.mount("/content/drive")
    # 체크포인트를 Drive에 "직접" 쓰지 않는다.
    #   · FUSE는 optimizer.pt 같은 대용량 파일 쓰기가 느리고, 도중에 세션이 끊기면
    #     반쯤 쓰인 파일이 남아 다음 세션의 resume이 통째로 실패한다.
    #   · 그래서 로컬 디스크에 쓰고, 저장이 "끝난 뒤에" 통째로 Drive에 복사한다(아래 콜백).
    OUTPUT_DIR = f"/content/{RUN}"
    SYNC_DIR = os.environ.get("SYNC", f"/content/drive/MyDrive/math_rft/{RUN}")
    DATA = DATA_ENV or "/content/drive/MyDrive/deep_chal_math_clean_v5.csv"
    MERGE_DIR = f"/content/{RUN}_merged"           # 드라이브 직접 저장은 0byte 나옴

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                    # 16 → 32
    lora_alpha=64,           # 32 → 64
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

df = pd.read_csv(DATA).sample(frac=1, random_state=42).reset_index(drop=True)

# 학습률 검증용 부분집합. 셔플 후 앞에서 자르므로 source 구성 비율이 그대로 유지된다.
if SUBSET and SUBSET < len(df):
    df = df.iloc[:SUBSET].reset_index(drop=True)
    print(f"[SUBSET] {SUBSET:,}개만 사용 (학습률 검증용 단축 실행)")

print(f"[설정] LR={LEARNING_RATE:g}  EPOCHS={EPOCHS:g}  데이터={DATA}")

# MAX_SEQ_LENGTH를 넘는 샘플은 학습에서 제외한다.
# 자르면 \boxed{} 없이 끝나는 샘플이 되어 "답을 안 내놓는 습관"을 다시 학습시킨다.
n_before = len(df)
texts = df["text"].tolist()
tok_len = []
for i in range(0, len(texts), 512):                    # 배치 토크나이즈 (40k면 한참 걸린다)
    tok_len += [len(x) for x in tokenizer(texts[i:i + 512])["input_ids"]]
df = df[[n <= MAX_SEQ_LENGTH for n in tok_len]].reset_index(drop=True)
kept = [n for n in tok_len if n <= MAX_SEQ_LENGTH]
print(f"학습 데이터 {len(df):,}개 (길이 초과로 제외 {n_before - len(df):,}개)")
print(df["source"].value_counts().to_string())

# 12시간 세션에 들어오는지 미리 계산. 넘으면 지금 데이터를 줄이는 게 낫다.
# 처리량 648 tok/s 는 T4 + unsloth padding-free 실측값 (2,515 step × 12.7 s/it).
BATCH, ACCUM = 2, 8
TOTAL_STEPS = -(-len(df) // (BATCH * ACCUM)) * EPOCHS
total_tok = sum(kept) * EPOCHS
print(f"\n총 {TOTAL_STEPS:,} step / {total_tok/1e6:.1f}M 토큰"
      f"  →  T4 648tok/s 실측 기준 약 {total_tok/648/3600:.1f}시간")
print("  ※ 학습 시작 후 20 step쯤에 진행바 ETA를 꼭 확인하세요.")
print("    11시간을 넘으면 즉시 중단하세요 (캐글 세션 한도 12시간).")
print("    09_build_train_mix.py 의 --external 을 줄여서 데이터를 다시 만들면 됩니다.")

dataset = Dataset.from_pandas(df[["text"]])

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    # 배치 2 × seq 1536 = 3,072토큰/스텝. T4를 포화시키기엔 이미 충분해서
    # 4로 올려도 속도는 10~20%만 늘고 활성값 메모리는 두 배가 된다.
    # Qwen 어휘가 151k라 logits이 크고(4×1536×151936), 여기에 group_by_length가
    # 긴 샘플끼리 배치를 몰아주므로 최악 배치가 반드시 온다.
    # 6시간 뒤 OOM으로 날리는 것보다 10% 느린 게 낫다.
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,      # effective batch 16
    learning_rate=LEARNING_RATE,            # 위 LEARNING_RATE 주석 참조. QLoRA 표준은 2e-4
    num_train_epochs=EPOCHS,
    # group_by_length 제거: TRL 0.24의 SFTConfig가 안 받아서 무시된다.
    # 대신 unsloth가 padding-free를 자동으로 켜는데, 패딩을 줄이는 게 아니라
    # 아예 없애므로 그쪽이 더 낫다. 그냥 unsloth에 맡긴다.
    warmup_steps=max(20, int(0.03 * TOTAL_STEPS)),   # warmup_ratio는 transformers 5.x에서 deprecated
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    optim="adamw_8bit",
    weight_decay=0.01,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    seed=42,
    report_to="none",
    # TRL 0.20부터 SFTConfig의 max_seq_length가 max_length로 이름이 바뀌었다.
    # 버전을 가리지 않도록 지원하는 쪽 이름으로 넘긴다.
    **pick(["max_seq_length", "max_length"], SFTConfig, MAX_SEQ_LENGTH),
)


# 커스텀 저장 콜백은 쓰지 않는다.
#   기존 노트북의 SafeSaveCallback은 model.save_pretrained()로 "LoRA 어댑터 가중치만" 저장했다.
#   거기엔 optimizer 상태 / LR 스케줄러 / 데이터 순서 / RNG가 없다. 그걸로 재시작하면
#   이어하기가 아니라 "그 가중치에서 새 학습 시작"이라, cosine LR이 처음 값으로 튀어오르고
#   Adam 모멘텀이 날아가서 모델이 오히려 망가진다.
#   게다가 저장 경로가 HF Trainer의 checkpoint-{step}과 똑같아서 진짜 체크포인트를 덮어쓴다.
#   → save_strategy="steps"(위 SFTConfig)가 optimizer/scheduler/sampler까지 전부 저장하므로
#     그것만 쓰면 된다. 캐글은 /kaggle/working 자체가 영구 Output이라 별도 복사도 불필요.

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    # 구 TRL은 tokenizer=, 신 TRL은 processing_class=. train_on_responses_only가
    # 트레이너에서 토크나이저를 꺼내 쓰므로 반드시 전달돼야 한다.
    **pick(["processing_class", "tokenizer"], SFTTrainer, tokenizer),
)

# 질문 부분에는 loss를 걸지 않는다 (기존 코드에서 이미 잘 하고 있던 부분 — 유지)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)


def check_masking(trainer, tok, n=4):
    """train_on_responses_only 가 실제로 걸렸는지 확인.

    이게 조용히 실패하면 시스템 프롬프트와 문제 지문에까지 loss가 걸린다.
    학습은 멀쩡히 도는 것처럼 보이는데 9시간 뒤 결과만 나빠지므로,
    시작 전에 눈으로 확인하고 넘어간다.
    """
    try:
        rows = [trainer.train_dataset[i] for i in range(n)]
        batch = trainer.data_collator(rows)
        labels, ids = batch["labels"], batch["input_ids"]
        flat_l, flat_i = labels.reshape(-1), ids.reshape(-1)
        keep = flat_l != -100
        ratio = keep.sum().item() / keep.numel()
        trained_txt = tok.decode(flat_i[keep][:60])
        print("\n[마스킹 점검]")
        print(f"  loss가 걸리는 토큰 비율: {ratio:.1%}")
        print(f"  loss 대상 앞부분: {trained_txt[:160]!r}")
        if ratio > 0.95:
            print("  ⚠️ 거의 전부에 loss가 걸립니다 → 마스킹 실패."
                  " 지문까지 학습하므로 중단하고 확인하세요.")
        elif "Please reason step by step" in trained_txt:
            print("  ⚠️ 시스템 프롬프트가 loss 대상에 들어있습니다 → 마스킹 실패.")
        else:
            print("  ✅ 정상 — 풀이 부분에만 loss가 걸립니다.")
    except Exception as e:
        print(f"\n[마스킹 점검] 확인 실패({type(e).__name__}: {e}) — 그대로 진행합니다.")


check_masking(trainer, tokenizer)


# ══════════════════════════════════════════════════════════
# 세션이 끊겨도 진도를 잃지 않게 — 저장 직후 Drive로 체크포인트 복사
# ══════════════════════════════════════════════════════════
# 코랩 무료 세션은 예고 없이 끊긴다. save_steps=100 마다 로컬에 저장된 체크포인트를
# 저장이 "완료된 뒤에" Drive로 옮겨두면, 다음 세션(다른 계정이어도)이 그 지점에서
# 이어받는다. optimizer/scheduler/sampler 상태까지 들어 있으므로 진짜 이어하기다.
#
# 복사는 임시 폴더에 받은 뒤 rename 한다. 복사 도중 세션이 죽어도 반쯤 쓰인 폴더가
# 정식 이름을 갖지 않으므로, 다음 세션이 그걸 골라 resume 하다 터지는 일이 없다.
if not ON_KAGGLE:
    from transformers import TrainerCallback   # noqa: E402

    class DriveSync(TrainerCallback):
        def __init__(self, dst, keep=2):
            self.dst, self.keep = dst, keep
            os.makedirs(dst, exist_ok=True)

        def on_save(self, args, state, control, **kw):
            src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(src):
                return
            final = os.path.join(self.dst, os.path.basename(src))
            tmp = final + ".partial"
            try:
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.copytree(src, tmp)
                shutil.rmtree(final, ignore_errors=True)
                os.rename(tmp, final)
                print(f"[DriveSync] {final}")
            except Exception as e:
                # 동기화 실패가 학습을 죽이면 안 된다. 다음 저장 때 다시 시도된다.
                print(f"[DriveSync] 실패 — 학습은 계속합니다 ({type(e).__name__}: {e})")
                return
            # Drive 용량을 위해 오래된 것 정리
            cks = sorted(glob.glob(os.path.join(self.dst, "checkpoint-*")),
                         key=lambda q: int(q.rsplit("-", 1)[1]))
            for old in cks[:-self.keep]:
                shutil.rmtree(old, ignore_errors=True)

    trainer.add_callback(DriveSync(SYNC_DIR))
    print(f"체크포인트 동기화 대상: {SYNC_DIR}")


def prepare_resume(output_dir, prev_run):
    """이어하기 준비. 재개할 게 있으면 True, 없으면 None(=처음부터) 반환.

    캐글에서 이전 세션을 이어받는 절차:
      1) 이전 노트북 실행이 끝나면 Output 탭 → Create Dataset
      2) 새 노트북에 그 Dataset을 Input으로 추가
      3) 아래 PREV_RUN을 그 경로로 지정
    /kaggle/input 은 읽기 전용이라 Trainer가 쓸 수 없으므로 working으로 복사해야 한다.
    """
    def latest(d):
        # ".partial" 은 복사가 끝나지 않은 폴더다. 절대 고르면 안 된다.
        cks = [c for c in glob.glob(os.path.join(d, "checkpoint-*"))
               if os.path.isdir(c) and c.rsplit("-", 1)[1].isdigit()]
        return max(cks, key=lambda p: int(p.rsplit("-", 1)[1])) if cks else None

    os.makedirs(output_dir, exist_ok=True)
    if latest(output_dir):
        print(f"이어하기: {latest(output_dir)}")
        return True

    if prev_run and os.path.isdir(prev_run):
        src = latest(prev_run) or latest(os.path.join(prev_run, os.path.basename(output_dir)))
        if src:
            dst = os.path.join(output_dir, os.path.basename(src))
            shutil.copytree(src, dst, dirs_exist_ok=True)
            if not os.path.exists(os.path.join(dst, "optimizer.pt")):
                print("경고: optimizer.pt가 없습니다. 어댑터만 있는 체크포인트라 "
                      "진짜 이어하기가 아니라 LR/모멘텀이 초기화됩니다.")
            print(f"이전 세션 체크포인트 복사: {src} → {dst}")
            return True

    print("체크포인트 없음 → 처음부터 학습")
    return None


# 이어하기 소스. 캐글은 Input 데이터셋 경로, 코랩은 Drive의 SYNC_DIR 을 준다.
#   PREV_RUN=/content/drive/MyDrive/math_rft/qwen_v5 python 03_train_sft.py
PREV_RUN = os.environ.get("PREV_RUN") or (None if ON_KAGGLE else globals().get("SYNC_DIR"))

trainer.train(resume_from_checkpoint=prepare_resume(OUTPUT_DIR, PREV_RUN))

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ══════════════════════════════════════════════════════════
# [CELL 3] 16bit 병합 (vLLM이 바로 읽는 포맷)
# ══════════════════════════════════════════════════════════
model.save_pretrained_merged(MERGE_DIR, tokenizer, save_method="merged_16bit")

if ON_KAGGLE:
    # /kaggle/working 자체가 노트북 Output이다.
    # 실행 끝난 뒤 노트북 우측 Output 탭 → "Create Dataset" 하면
    # 추론 노트북에서 --model /kaggle/input/<이름>/qwen_v4_merged 로 바로 쓸 수 있다.
    # 용량 절약: 어댑터 체크포인트는 병합 끝났으면 지운다 (Output 20GB 제한)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    print(f"완료 → {MERGE_DIR}  (Output 탭에서 Create Dataset)")
else:
    # 드라이브에 직접 병합 저장하면 FUSE 때문에 0byte가 나온다 (기존 세션에서 겪은 문제).
    # 로컬에 저장 → 압축 → 드라이브로 복사.
    shutil.make_archive(f"/content/{RUN}_ready", "zip", MERGE_DIR)
    shutil.copy(f"/content/{RUN}_ready.zip", f"/content/drive/MyDrive/{RUN}_ready.zip")
    print(f"완료 → /content/drive/MyDrive/{RUN}_ready.zip")
