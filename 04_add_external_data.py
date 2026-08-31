"""
04_add_external_data.py  —  공개 데이터셋 추가 (0.8 돌파를 위한 핵심 레버)

정제 후 학습 데이터가 8.2k로 줄었다. 3B 모델의 수학 추론을 제대로 올리려면 30~50k가 필요하다.
대회 규칙 5.2: "공개 데이터셋의 추가 사용은 자유" + 5.2c "사용한 외부 데이터셋 목록 명시 필수".

쓰는 데이터셋 (둘 다 무료 공개, 상업적 이용 가능 라이선스):
  · nvidia/OpenMathInstruct-2   (CC-BY-4.0)  — GSM8K/MATH 증강, 풀이가 짧고 항상 \boxed{} 종료,
                                               expected_answer로 자체 검증되어 있어 품질이 균일
  · AI-MO/NuminaMath-CoT        (Apache-2.0) — cn_k12/올림피아드 계열. 리더보드에 이 계열이
                                               섞여 있으므로(LaTeX 문항 44%) 도메인이 맞는다

주의: 정답이 정수인 문항만 남긴다 (대회 채점이 정수 exact match).
      리더보드/로컬검증 문항과 겹치는 건 전부 제거한다.

Colab에서 실행 (다운로드 용량이 크므로 로컬 말고 Colab 권장):
    !pip install -q datasets
    !python 04_add_external_data.py
"""
import argparse
import glob
import os
import re

import pandas as pd
from datasets import load_dataset

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def resolve(name):
    """입력 파일 찾기.

    캐글은 데이터셋마다 마운트 경로가 달라서(/kaggle/input/<owner>/<slug>/...) 상대경로가
    거의 항상 깨진다. 그래서 순서대로 뒤진다:
      1) 준 경로 그대로   2) 현재 작업 디렉터리
      3) 이 스크립트가 있는 폴더 (보통 csv가 같은 데이터셋에 같이 올라가 있음)
      4) /kaggle/input 전체를 파일명으로 재귀 탐색
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [name, os.path.join(os.getcwd(), name), os.path.join(here, name)]:
        if os.path.isfile(c):
            return c
    hits = sorted(glob.glob(f"/kaggle/input/**/{os.path.basename(name)}", recursive=True))
    if hits:
        print(f"  자동 탐색: {os.path.basename(name)} → {hits[0]}")
        return hits[0]
    raise FileNotFoundError(
        f"'{name}' 을 찾을 수 없습니다.\n"
        f"  찾아본 곳: 현재 폴더({os.getcwd()}), 스크립트 폴더({here}), /kaggle/input 전체\n"
        f"  → --base / --val / --lb 옵션으로 절대경로를 직접 넣어주세요."
    )

# 개수는 "12시간(캐글 세션 상한) 안에 1 epoch이 끝나는가"로 정한다.
#   T4 QLoRA 3B 실효 처리량 ≈ 800 tok/s → 12h ≈ 34M 토큰
#   대회 데이터 8.2k × 1,150tok = 9.4M
#   외부  32k × ~700tok        = 22.4M   (외부 풀이가 짧아서 같은 시간에 더 많이 본다)
#   합계 약 32M → 약 11시간. 들어온다.
# 20k×2epoch 이 아니라 40k×1epoch 을 하는 이유: 비용은 같은데 데이터 다양성이 두 배다.
N_OPENMATH = 25000   # GSM8K/MATH 증강. 짧고 항상 \boxed{} 종료
N_NUMINA = 7000      # cn_k12/올림피아드 계열 (리더보드 LaTeX 문항 44% 대응)
MAX_SOL_CHARS = 2500


def norm_q(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def last_boxed(text):
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
    return None


def as_int(x):
    if x is None:
        return None
    s = str(x).replace(",", "").replace(" ", "").replace("$", "")
    try:
        f = float(s)
    except ValueError:
        return None
    return int(round(f)) if abs(f - round(f)) < 1e-9 and abs(f) < 1e12 else None


def build(q, sol):
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n{sol.strip()}<|im_end|>")


def collect(rows, banned, seen):
    """rows: (question, solution) iterable → 검증 통과한 레코드 리스트"""
    out = []
    for q, sol in rows:
        if not q or not sol or len(sol) > MAX_SOL_CHARS:
            continue
        if sol.count("\\boxed{") != 1:
            continue
        a = as_int(last_boxed(sol))
        if a is None:
            continue
        k = norm_q(q)
        if k in banned or k in seen:
            continue
        seen.add(k)
        out.append({"id": f"ext-{len(seen)}", "question": q, "answer": a,
                    "source": "external", "text": build(q, sol)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deep_chal_math_clean_v4.csv")
    ap.add_argument("--val", default="local_val_500.csv")
    ap.add_argument("--lb", default="deep_chal_math_leaderboard_filtered.csv")
    ap.add_argument("--out", default=None, help="기본값: 캐글이면 /kaggle/working, 아니면 현재 폴더")
    ap.add_argument("--n-openmath", type=int, default=N_OPENMATH)
    ap.add_argument("--n-numina", type=int, default=N_NUMINA)
    a = ap.parse_args()

    print("입력 파일 확인:")
    base_p, val_p, lb_p = resolve(a.base), resolve(a.val), resolve(a.lb)
    for p in (base_p, val_p, lb_p):
        print(f"  {p}")

    # /kaggle/input 은 읽기 전용이라 출력은 반드시 working 아래로
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    out_p = a.out or os.path.join(out_dir, "deep_chal_math_clean_v5.csv")

    base = pd.read_csv(base_p)
    banned = set(pd.read_csv(val_p)["question"].map(norm_q)) | set(pd.read_csv(lb_p)["question"].map(norm_q))
    seen = set(base["question"].map(norm_q))
    print(f"기존 {len(base):,} / 제외 대상(평가·검증) {len(banned):,}")

    added = []

    # ── OpenMathInstruct-2 ────────────────────────────────────────────
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=True)
    buf = []
    for r in ds:
        buf.append((r["problem"], r["generated_solution"]))
        if len(buf) >= a.n_openmath * 3:
            break
    got = collect(buf, banned, seen)[:a.n_openmath]
    added += got
    print(f"OpenMathInstruct-2: {len(got):,}개 채택")

    # ── NuminaMath-CoT ────────────────────────────────────────────────
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    buf = []
    for r in ds:
        buf.append((r["problem"], r["solution"]))
        if len(buf) >= a.n_numina * 4:
            break
    got = collect(buf, banned, seen)[:a.n_numina]
    added += got
    print(f"NuminaMath-CoT: {len(got):,}개 채택")

    final = pd.concat([base, pd.DataFrame(added)], ignore_index=True)
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)
    final.to_csv(out_p, index=False)
    print(f"\n최종 {len(final):,}개 → {out_p}")
    print(final["source"].value_counts().to_string())
    print("\n※ 최종 제출 시 방법론 문서에 아래 데이터셋 출처를 반드시 명시할 것 (규칙 5.2c)")
    print("   - nvidia/OpenMathInstruct-2 (CC-BY-4.0)")
    print("   - AI-MO/NuminaMath-CoT (Apache-2.0)")
    print("   - openai/gsm8k (MIT)")


if __name__ == "__main__":
    main()
