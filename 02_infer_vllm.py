"""
02_infer_vllm.py  —  vLLM Self-Consistency 추론 (기존 캐글추론코드.py 대체)

기존 코드 대비 변경점과 이유:
  1) repetition_penalty 1.15 → 1.0
     수학 풀이는 같은 숫자/변수를 반복해서 써야 한다. 1.15는 두 번째 등장하는 숫자의
     확률을 깎아 계산 자체를 망가뜨린다. 앵무새 증상은 학습 데이터에서 고쳐야 할 문제
     (01_clean_data.py)이지 추론에서 억제할 문제가 아니다.
  2) "but be concise" 제거 — CoT를 짧게 만들면 정확도가 떨어진다.
  3) temperature 0.0 단일 샘플 → n=8~16 다수결 (Self-Consistency).
     vLLM은 SamplingParams(n=K)로 프롬프트 KV 캐시를 공유하므로 K배보다 훨씬 싸다.
  4) max_model_len 8192 → 2048.
     max_model_len은 시퀀스당 KV 캐시 예약량을 결정한다. 8192는 동시 처리 시퀀스 수를
     1/4로 줄여서 처리량을 크게 떨어뜨린다. 프롬프트 ~300토큰 + 출력 1024면 충분.
  5) 정답 추출기 전면 교체.
     기존 "텍스트 전체의 마지막 숫자" fallback은 중간 계산값을 집어오기 쉬워 매우 위험.
     \boxed{} → 꼬리 문장의 "answer is N" → 마지막 '줄'의 숫자 순으로 단계적 폴백하고,
     \boxed{}를 낸 샘플이 하나라도 있으면 그 표들만 가지고 투표한다.
  6) 2×T4 데이터 병렬 옵션 (--shard). tensor_parallel_size=2 는 /dev/shm 문제로 터졌지만,
     GPU마다 독립 프로세스를 띄워 문제를 반으로 나누면 NCCL 없이 2배 속도가 나온다.

사용법
  단일 GPU:
      python 02_infer_vllm.py --input test.csv --output submission.csv --n 8
  로컬 검증(정확도까지 출력):
      python 02_infer_vllm.py --input local_val_500.csv --output val_pred.csv --n 8
  2×T4 데이터 병렬 (노트북 셀에서):
      import subprocess, os
      ps = [subprocess.Popen(
                ["python", "02_infer_vllm.py", "--input", TEST, "--output", f"sub_{i}.csv",
                 "--shard", str(i), "--num-shards", "2", "--n", "16"],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i in range(2)]
      [p.wait() for p in ps]
      import pandas as pd
      pd.concat([pd.read_csv(f"sub_{i}.csv") for i in range(2)]).to_csv("submission.csv", index=False)
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import Counter

# ── vLLM 엔진 설정 ─────────────────────────────────────────────────────
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# VLLM_USE_V1=0 은 일부러 설정하지 않는다.
#   · 예전에 V0을 강제한 이유는 tensor_parallel_size=2 의 NCCL/shm 충돌 때문이었는데,
#     지금은 GPU당 독립 프로세스(--shard)라 NCCL 자체를 안 쓴다.
#   · vLLM 0.11+ 에서는 V0 엔진이 제거돼서 강제하면 오히려 실행이 안 된다.
#   문제가 생기면 실행 전에 export VLLM_USE_V1=0 으로 직접 덮어쓰면 된다.

import pandas as pd  # noqa: E402

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
# 프롬프트 앙상블용 변형. temperature 는 같은 사고 틀 안에서 토큰만 흔들지만,
# 프롬프트는 문제를 바라보는 틀 자체를 바꾼다 —
# 다수결이 틀리는 원인인 "공유된 논리 오류"를 깨려면 후자가 필요하다.
#   · 넷 다 \\boxed{} 를 유지한다. 이걸 빼면 답 추출이 무너져 절단 처리로 떨어진다.
PROMPT_VARIANTS = {
    "default": SYSTEM_PROMPT,
    "formula": (
        "First state the relevant formula or definition, then substitute the "
        "given values and compute. Put your final answer within \\boxed{}."
    ),
    "check": (
        "Solve the problem step by step. Before stating the final answer, "
        "verify it by substituting back or by solving a second way. "
        "Put your final answer within \\boxed{}."
    ),
    "restate": (
        "Restate what the problem asks and list the given quantities, then "
        "solve step by step. Put your final answer within \\boxed{}."
    ),
}
MAX_ABS = 10 ** 15  # 이 이상은 환각. 정답 분포상 |answer| > 10^9 는 0.1% 뿐이다.


# ══════════════════════════════════════════════════════════════════
# 정답 추출
# ══════════════════════════════════════════════════════════════════
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FRAC = re.compile(r"\\[dt]?frac\s*\{\s*(-?[\d.]+)\s*\}\s*\{\s*(-?[\d.]+)\s*\}")
_ANSWER_TAIL = re.compile(
    r"(?:answer|answer is|result is|equals|therefore|so,?)\D{0,20}?(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


def last_boxed(text: str):
    """중첩 중괄호를 세면서 마지막 \boxed{...} 내용을 반환. 잘린 출력도 최대한 건짐."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return text[j + 1:]  # max_tokens에 잘려 닫는 괄호가 없는 경우


def to_int(s):
    """LaTeX 조각을 정수로. 실패하면 None (0을 반환하지 않는 게 중요 — 투표에서 빠져야 함)"""
    if s is None:
        return None
    s = str(s).strip()

    m = _FRAC.search(s)
    if m:
        try:
            num, den = float(m.group(1)), float(m.group(2))
            if den == 0:
                return None
            return _finish(num / den)
        except ValueError:
            pass

    s = re.sub(r"\\(?:text|mathrm|mbox|textbf|textit)\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)                 # \! \, \; \pi 등 잔여 커맨드 제거
    s = s.replace("{", " ").replace("}", " ").replace("$", "").replace("%", "")
    s = s.replace(",", "").replace(" ", "")

    try:
        return _finish(float(s))
    except ValueError:
        pass

    nums = _NUM.findall(s)
    if not nums:
        return None
    try:
        return _finish(float(nums[-1].replace(",", "")))
    except ValueError:
        return None


def _finish(v: float):
    if v != v or v in (float("inf"), float("-inf")):  # NaN/Inf
        return None
    if abs(v) > MAX_ABS:
        return None
    return int(round(v))


def extract(text: str):
    """(정답, 신뢰도등급) 반환. 등급 2=\boxed, 1=꼬리문장, 0=마지막줄"""
    b = last_boxed(text)
    if b is not None:
        v = to_int(b)
        if v is not None:
            return v, 2

    tail = text[-500:]
    m = _ANSWER_TAIL.findall(tail)
    if m:
        v = to_int(m[-1])
        if v is not None:
            return v, 1

    # 마지막 '줄'의 숫자만. 텍스트 전체의 마지막 숫자를 쓰면 중간 계산값을 집어온다.
    for line in reversed([l for l in text.strip().splitlines() if l.strip()]):
        nums = _NUM.findall(line)
        if nums:
            v = to_int(nums[-1])
            if v is not None:
                return v, 0
    return None, -1


def vote_from_pairs(pairs):
    """등급이 가장 높은(=\boxed를 낸) 샘플들끼리만 다수결. 동점이면 먼저 나온 답.

    (값, 등급) 목록을 그대로 받는다. extract()를 호출부에서 한 번만 돌리고
    진단·투표·RFT 저장이 같은 결과를 재사용하기 위한 형태.
    반환: (답, 합의율, {답: 표수})  ← 표 분포까지 돌려주므로 여러 실행을 정확히 합칠 수 있다.
    """
    cands = [(v, g) for v, g in pairs if v is not None]
    if not cands:
        return 0, 0.0, {}
    best_grade = max(g for _, g in cands)
    votes = [v for v, g in cands if g == best_grade]
    counter = Counter(votes)
    top = counter.most_common(1)[0][1]
    win = next(v for v in votes if counter[v] == top)   # 동점 시 등장 순서가 빠른 쪽
    return win, top / len(pairs), {str(k): c for k, c in counter.items()}


def majority_vote(texts):
    """텍스트에서 바로 투표. 기존 호출부 호환용."""
    return vote_from_pairs([extract(t) for t in texts])[:2]


# ══════════════════════════════════════════════════════════════════
# 인자 검증 (엔진을 띄우기 전에 확실히 죽인다)
#   vLLM은 --model에 뭐가 들어와도 일단 로드를 시도해서, 잘못 넣으면
#   "config file is not a valid JSON" 같은 엉뚱한 40줄 트레이스백이 나온다.
# ══════════════════════════════════════════════════════════════════
def check_paths(args):
    def die(msg):
        print(f"\n[설정 오류]\n{msg}\n", file=sys.stderr)
        raise SystemExit(2)

    m = args.model
    if os.path.isfile(m):
        die(f"--model 에 '파일'이 들어왔습니다: {m}\n"
            f"  --model 은 config.json 이 들어있는 '모델 폴더'이거나\n"
            f"  HF 이름(예: Qwen/Qwen2.5-3B-Instruct)이어야 합니다.\n"
            f"\n"
            f"  ※ CSV를 넣으셨다면 학습을 하려던 것 같습니다.\n"
            f"     학습은 이 파일(02, 추론)이 아니라 03_train_sft.py 입니다.")

    if os.path.isdir(m):
        if not os.path.isfile(os.path.join(m, "config.json")):
            sub = glob.glob(os.path.join(m, "*", "config.json"))
            hint = f"\n  혹시 이 경로 아닌가요? → {os.path.dirname(sub[0])}" if sub else ""
            die(f"--model 폴더에 config.json 이 없습니다: {m}{hint}")
    elif m.startswith("/") or m.startswith("./"):
        die(f"--model 경로가 존재하지 않습니다: {m}")
    # 그 외(슬래시 포함 이름)는 HF 저장소로 보고 통과 — 다운로드는 vLLM에 맡긴다

    if not os.path.isfile(args.input):
        die(f"--input CSV 를 찾을 수 없습니다: {args.input}")
    # 출력 폴더는 여기서 만든다. 없으면 20분 생성한 뒤 첫 저장에서 죽는다 —
    # to_csv 는 부모 디렉터리를 만들어 주지 않는다.
    for d in {os.path.dirname(os.path.abspath(args.output)),
              os.path.abspath(args.backup_dir) if args.backup_dir else None,
              os.path.dirname(os.path.abspath(args.dump_gens)) if args.dump_gens else None}:
        if d:
            os.makedirs(d, exist_ok=True)


    # GPU 확인. 없으면 vLLM이 "Device string must not be empty" 같은
    # 원인을 알 수 없는 에러를 내므로 여기서 먼저 잡는다.
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        die("GPU가 잡히지 않습니다. vLLM은 CPU로 돌릴 수 없습니다.\n"
            "  · Colab  : 런타임 → 런타임 유형 변경 → T4 GPU → 저장\n"
            "             (런타임을 바꾸면 /content가 초기화되니 다운로드 셀부터 다시 실행)\n"
            "  · Kaggle : Settings → Accelerator → GPU T4 x2\n"
            "             GPU 쿼터가 남아 있는지도 확인하세요.")
    print(f"GPU: {torch.cuda.get_device_name(0)} × {torch.cuda.device_count()}")


# ══════════════════════════════════════════════════════════════════
# vLLM 엔진 생성 (버전 호환)
# ══════════════════════════════════════════════════════════════════
def build_llm(LLM, args):
    import inspect

    try:
        import vllm
        print(f"vLLM {vllm.__version__}")
    except Exception:
        pass

    want = {
        "model": args.model,
        "tensor_parallel_size": 1,      # 2 GPU는 --shard 데이터 병렬로 (NCCL 회피)
        "dtype": "half",                # T4는 bf16 미지원
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_util,
        "enforce_eager": not args.no_eager,
        "swap_space": 2,                # 구버전에만 존재 → 없으면 자동으로 빠짐
        "seed": args.seed,
        "trust_remote_code": True,
    }

    # 실제 LLM(**kwargs)는 EngineArgs로 흘러가므로 EngineArgs가 권위 있는 목록이다.
    # EngineArgs를 못 읽으면 필터링을 아예 하지 않는다 — 어설프게 걸렀다가 dtype이나
    # max_model_len이 조용히 빠지면 T4에서 bf16/8192로 잘못 돌아가고 아무도 눈치채지 못한다.
    # 그 경우엔 아래 TypeError 재시도 루프에만 의존한다 (터지면서 알려주는 게 낫다).
    supported = set()
    try:
        from vllm.engine.arg_utils import EngineArgs
        supported |= set(inspect.signature(EngineArgs.__init__).parameters)
        supported |= set(getattr(EngineArgs, "__dataclass_fields__", {}))
    except Exception:
        supported = set()

    if supported:
        try:
            sig = inspect.signature(LLM.__init__)
            if not any(pp.kind is inspect.Parameter.VAR_KEYWORD
                       for pp in sig.parameters.values()):
                supported |= set(sig.parameters)   # **kwargs 없으면 LLM 시그니처도 합집합
        except Exception:
            pass
        dropped = [k for k in want if k not in supported and k != "model"]
        for k in dropped:
            want.pop(k)
        if dropped:
            print(f"이 vLLM 버전이 지원하지 않아 제외한 인자: {dropped}")

    # 시그니처 검사로도 못 거른 인자는 에러 메시지를 보고 하나씩 빼면서 재시도
    for _ in range(len(want)):
        try:
            return LLM(**want)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m or m.group(1) not in want:
                raise
            bad = m.group(1)
            print(f"'{bad}' 미지원 → 제외하고 재시도")
            want.pop(bad)
    raise RuntimeError("vLLM 엔진 초기화 실패")


# ══════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/kaggle/input/qwen-math-v4/qwen2.5_3b_math_v4_merged")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--n", type=int, default=8, help="Self-Consistency 샘플 수")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    # 1024는 실측에서 19.3%가 잘렸다(= \boxed{} 쓰기 전에 끊김). 2048로 올린다.
    # max_model_len 은 프롬프트+출력을 모두 담아야 하므로 그보다 커야 한다.
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=3072)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--gpu-util", type=float, default=0.90)
    p.add_argument("--no-eager", action="store_true",
                   help="CUDA graph 사용 (~15%% 빠름). OOM 나면 끄세요")
    p.add_argument("--backup-dir", default=None,
                   help="청크마다 결과를 복사해둘 경로 (예: /content/drive/MyDrive/math_results). "
                        "세션이 죽어도 결과가 남고, 재실행 시 자동으로 복구해서 이어합니다")
    # ── RFT(거절 샘플링 자기학습) 데이터 수집용 ─────────────────────────
    p.add_argument("--dump-gens", default=None,
                   help="생성된 풀이 원문을 JSONL 조각 파일로 저장할 디렉터리. "
                        "정답 컬럼이 있는 입력에서는 '정답과 일치한 풀이'만 저장한다 (RFT 학습데이터).")
    p.add_argument("--keep-max", type=int, default=4,
                   help="--dump-gens 시 문항당 저장할 최대 풀이 수 (완전중복 제거 후)")
    p.add_argument("--dump-all", action="store_true",
                   help="--dump-gens 시 오답 풀이도 함께 저장 (오답 분석용, 용량 크게 증가)")
    p.add_argument("--seed", type=int, default=1234,
                   help="샘플링 시드. 같은 시드로 두 번 돌리면 거의 같은 결과가"
                        " 나오므로, 표본만 늘리려면 반드시 다르게 줄 것")
    p.add_argument("--system", default="default",
                   help="시스템 프롬프트: PROMPT_VARIANTS 의 이름"
                        " (default/formula/check/restate) 또는 직접 쓴 문장")
    args = p.parse_args()
    check_paths(args)          # vLLM 로드 전에 검증

    from vllm import LLM, SamplingParams

    dbg = args.output.replace(".csv", "_debug.csv")

    # 구글 드라이브에 직접 append 하면 FUSE 때문에 파일이 깨질 수 있다.
    # 로컬에 쓰고 청크마다 복사하는 방식을 쓴다. 복사 실패가 4시간짜리 실행을
    # 죽이면 안 되므로 예외는 삼킨다.
    def nlines(path):
        try:
            with open(path, encoding="utf-8") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def backup():
        if not args.backup_dir:
            return
        try:
            os.makedirs(args.backup_dir, exist_ok=True)
            for p_ in (args.output, dbg):
                if os.path.isfile(p_):
                    dst_ = os.path.join(args.backup_dir, os.path.basename(p_))
                    # 백업이 로컬보다 길면 덮어쓰지 않는다. 같은 tag 를 두 계정에서 돌리면
                    # 짧은 쪽이 긴 쪽을 지워버린다 — 실제로 한 번 날렸다.
                    if nlines(dst_) > nlines(p_):
                        print(f"  [백업 건너뜀] 백업이 더 깁니다 "
                              f"({nlines(dst_)} > {nlines(p_)}행). 같은 tag 를 다른 계정에서 "
                              f"돌리고 있지 않은지 확인하세요: {dst_}")
                        continue
                    shutil.copy(p_, dst_)
            # 생성 원문은 수백 MB까지 커지므로 매 청크 전체 복사는 감당이 안 된다.
            # 청크별 조각 파일 중 "아직 안 올라간 것"만 골라서 복사한다.
            # 여러 계정이 같은 백업 폴더를 보고 릴레이로 돌려도 조각 이름이 id 기반이라 안 겹친다.
            if args.dump_gens:
                gdst = os.path.join(args.backup_dir, "gens")
                os.makedirs(gdst, exist_ok=True)
                for src in glob.glob(os.path.join(args.dump_gens, "gens_*.jsonl")):
                    dst = os.path.join(gdst, os.path.basename(src))
                    if not os.path.isfile(dst):
                        shutil.copy(src, dst)
        except Exception as e:
            print(f"  [백업 실패, 계속 진행] {type(e).__name__}: {e}")

    # 이전 세션이 남긴 결과가 백업에 있으면 먼저 되살린다 (이어하기 판정 전에)
    if args.backup_dir:
        for p_ in (args.output, dbg):
            src = os.path.join(args.backup_dir, os.path.basename(p_))
            if os.path.isfile(src) and not os.path.isfile(p_):
                os.makedirs(os.path.dirname(os.path.abspath(p_)) or ".", exist_ok=True)
                shutil.copy(src, p_)
                print(f"백업에서 복구: {src}")

    # debug CSV에 votes/n_correct 컬럼이 추가됐다. 구버전이 만든 파일에 그대로 append 하면
    # 헤더와 열 수가 어긋나 조용히 깨진다. 형식이 다르면 비켜놓고 새로 쓴다.
    # (이어하기 판정은 args.output 기준이므로 재개 자체에는 영향이 없다.)
    if os.path.isfile(dbg):
        with open(dbg, encoding="utf-8") as f:
            head = f.readline().strip().split(",")
        if "votes" not in head:
            os.replace(dbg, dbg + ".oldfmt")
            print(f"구버전 debug 파일 형식 감지 → {os.path.basename(dbg)}.oldfmt 로 이름 변경 후 새로 기록")

    df = pd.read_csv(args.input)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    if args.num_shards > 1:
        df = df.iloc[args.shard::args.num_shards].reset_index(drop=True)
        print(f"[shard {args.shard}/{args.num_shards}] 담당 {len(df)}문항")

    # 이어하기
    if os.path.exists(args.output):
        done = set(pd.read_csv(args.output)["id"].astype(str))
        df = df[~df["id"].astype(str).isin(done)]
        print(f"기존 {len(done)}개 완료 → 남은 {len(df)}개부터 재개")
    if len(df) == 0:
        print("이미 전부 완료됨")
        return

    # vLLM 버전마다 EngineArgs에서 인자가 사라진다 (예: 0.11+ 에서 swap_space 제거).
    # 시그니처를 직접 보고 지원되는 것만 넘겨서 TypeError를 원천 차단한다.
    llm = build_llm(LLM, args)

    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=1.0,        # ← 핵심 변경. 절대 1.0 초과 금지
    )

    sys_prompt = PROMPT_VARIANTS.get(args.system, args.system)
    if sys_prompt != SYSTEM_PROMPT:
        print(f"[프롬프트] {args.system}: {sys_prompt}")

    def build(q):
        return (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    has_gt = "answer" in df.columns and df["answer"].notna().all()
    n_correct = n_seen = 0

    # 다수결 표 수(n)를 몇으로 할지는 GPU 시간을 크게 좌우한다.
    # n=32로 한 번만 돌려도, 앞 k개만 써서 투표하면 n=1/2/4/8/16 결과를 공짜로 얻는다.
    # 굳이 n을 바꿔가며 여러 번 돌릴 필요가 없다.
    VOTE_K = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= args.n]
    correct_k = {k: 0 for k in VOTE_K}
    n_gen = n_trunc = 0          # max_tokens에 잘린 샘플 수
    n_done = n_done_boxed = 0    # 끝까지 생성된 샘플, 그중 \boxed 낸 것
    grade_hist = {2: 0, 1: 0, 0: 0, -1: 0}   # 2=\boxed, 1=꼬리문장, 0=마지막줄, -1=추출실패

    for i in range(0, len(df), args.chunk):
        part = df.iloc[i:i + args.chunk]
        outs = llm.generate([build(q) for q in part["question"]], sp)

        gt_vals = (pd.to_numeric(part["answer"], errors="coerce").values
                   if has_gt else [None] * len(part))
        preds, confs, truncs, vote_js, n_corrs = [], [], [], [], []
        gens_rows = []
        for row_i, (o, gt_v) in enumerate(zip(outs, gt_vals)):
            texts = [c.text for c in o.outputs]
            done_flags = [getattr(c, "finish_reason", None) != "length" for c in o.outputs]
            # 문항별 잘림 수. 어려운(=긴) 문제에 잘림이 몰리는지 확인용.
            truncs.append(sum(not d for d in done_flags))
            # extract()는 문항당 한 번만 돌리고 진단·투표·RFT 저장이 결과를 공유한다.
            # (기존에는 진단 루프와 majority_vote가 같은 텍스트를 두 번 파싱했다.)
            pairs = [extract(t) for t in texts]
            # 진단: 출력이 잘렸는지 / 정답을 어느 경로로 뽑았는지.
            # 잘린 샘플은 애초에 \boxed{}까지 도달하지 못하므로, 포맷 학습이 됐는지는
            # "끝까지 생성된 것 중 몇 %가 \boxed를 냈나"로 따로 봐야 한다.
            for (_v, g), done in zip(pairs, done_flags):
                n_gen += 1
                grade_hist[g] += 1
                if done:
                    n_done += 1
                    n_done_boxed += (g == 2)
                else:
                    n_trunc += 1
            v, c, counter = vote_from_pairs(pairs)
            preds.append(v)
            confs.append(c)
            # 표 분포를 그대로 남긴다. 여러 번 돌린 결과를 나중에 정확히 합칠 수 있다.
            # (agree×n 으로 근사하면 2등 이하 표가 사라져 합산이 틀어진다.)
            vote_js.append(json.dumps(counter, ensure_ascii=False))

            n_corr = (sum(int(_v == gt_v) for _v, _ in pairs if _v is not None)
                      if has_gt else -1)
            n_corrs.append(n_corr)

            # 같은 생성 결과에서 앞 k개만 써서 투표 → n별 정확도 곡선을 공짜로 얻는다
            if has_gt:
                for k in VOTE_K:
                    vk, _, _ = vote_from_pairs(pairs[:k])
                    correct_k[k] += int(vk == gt_v)

            # ── RFT: 정답과 일치한 풀이만 골라 학습데이터 후보로 저장 ──────────
            if args.dump_gens:
                sols, seen = [], set()
                for t, (_v, g), done in zip(texts, pairs, done_flags):
                    if not done:
                        continue           # 잘린 풀이는 \boxed까지 못 갔으므로 학습에 못 쓴다
                    ok = has_gt and _v is not None and _v == gt_v
                    if not ok and not args.dump_all:
                        continue
                    key = re.sub(r"\s+", " ", t).strip()
                    if key in seen:        # 토씨까지 같은 풀이는 중복
                        continue
                    seen.add(key)
                    sols.append({"text": t, "val": _v, "grade": g, "correct": bool(ok)})
                    if len(sols) >= args.keep_max:
                        break
                if sols:
                    gens_rows.append({
                        "id": str(part["id"].values[row_i]),
                        "question": str(part["question"].values[row_i]),
                        # to_numeric(errors="coerce")가 NaN을 낼 수 있다. int(NaN)은 예외라
                        # 몇 시간짜리 실행을 통째로 죽인다.
                        "answer": (int(gt_v) if gt_v is not None and gt_v == gt_v else None),
                        "n_correct": n_corr, "n": len(texts),
                        "solutions": sols,
                    })

        res = pd.DataFrame({"id": part["id"].values, "answer": preds,
                            "agree": confs, "n_trunc": truncs, "votes": vote_js})
        if has_gt:
            # 한 번도 못 맞힌 문항(n_correct==0)은 RFT로는 학습 신호를 못 만든다.
            # 그 목록이 곧 "Gemini로 풀이를 채워야 할 문항"이 된다.
            res["n_correct"] = n_corrs
        res[["id", "answer"]].to_csv(
            args.output, mode="a" if os.path.exists(args.output) else "w",
            header=not os.path.exists(args.output), index=False,
        )
        res.to_csv(dbg, mode="a" if os.path.exists(dbg) else "w",
                   header=not os.path.exists(dbg), index=False)

        # 조각 파일 이름은 청크 첫 id 기준. 인덱스 기준으로 하면 이어하기로 df가 줄었을 때
        # i가 0부터 다시 시작해 이전 조각을 덮어쓴다. id는 릴레이 중에도 안 겹친다.
        if args.dump_gens and gens_rows:
            os.makedirs(args.dump_gens, exist_ok=True)
            fn = os.path.join(args.dump_gens,
                              f"gens_s{args.shard}_{gens_rows[0]['id']}.jsonl")
            with open(fn, "w", encoding="utf-8") as f:
                for r in gens_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        backup()

        if has_gt:
            n_correct += sum(int(a == b) for a, b in zip(preds, gt_vals))
            n_seen += len(part)
            print(f"  {i+len(part):5d}/{len(df)}  누적 정확도 {n_correct/n_seen:.4f}"
                  f"  평균 합의율 {sum(confs)/len(confs):.2f}")
        else:
            print(f"  {i+len(part):5d}/{len(df)} 저장  평균 합의율 {sum(confs)/len(confs):.2f}")

    if has_gt:
        print(f"\n★ 최종 정확도: {n_correct/n_seen:.4f}  ({n_correct}/{n_seen})")
        if len(VOTE_K) > 1:
            print("\n[다수결 표 수(n)별 정확도] — 같은 생성 결과를 재사용한 값")
            prev = None
            for k in VOTE_K:
                acc = correct_k[k] / n_seen
                delta = f"  ({acc-prev:+.4f})" if prev is not None else ""
                print(f"  n={k:<3d} {acc:.4f}{delta}")
                prev = acc
            print("  → 증가폭이 +0.005 아래로 떨어지는 지점이 GPU 시간 대비 최적")

    # ── 진단 리포트 ────────────────────────────────────────────────
    # 잘림 비율이 높으면 --max-tokens 를 올려야 하고,
    # \boxed 비율이 낮으면 모델이 포맷을 못 지키는 것이라 학습 데이터 문제다.
    print("\n[진단]")
    tot = max(sum(grade_hist.values()), 1)
    print(f"  생성 샘플 {n_gen:,}개 중 max_tokens에 잘림: {n_trunc:,}개 ({n_trunc/max(n_gen,1):.1%})"
          f"  → 10% 넘으면 --max-tokens 올릴 것")
    print(f"  ★ 끝까지 생성된 것 중 \\boxed{{}} 비율: {n_done_boxed/max(n_done,1):6.1%}"
          f"   → 이게 포맷 학습 지표. 95% 넘으면 정상")
    print(f"     (전체 기준 \\boxed 비율 {grade_hist[2]/tot:.1%} 는 잘림에 희석되므로 보지 말 것)")
    print(f"  꼬리 문장 폴백       : {grade_hist[1]/tot:6.1%}")
    print(f"  마지막 줄 폴백       : {grade_hist[0]/tot:6.1%}   ← 대부분 잘린 샘플")
    print(f"  추출 실패            : {grade_hist[-1]/tot:6.1%}")
    backup()
    print(f"\n완료 → {args.output}")
    if args.backup_dir:
        print(f"       백업 → {os.path.join(args.backup_dir, os.path.basename(args.output))}")


if __name__ == "__main__":
    main()
