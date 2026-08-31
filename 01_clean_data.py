"""
01_clean_data.py  —  SFT 학습 데이터 정제 (가장 중요한 단계)

현재 deep_chal_math_master_train_v3.csv 의 실측 문제점:
  · competition_gemini_cot 16,359개 중 30.4%(4,968개)가 \boxed{} 를 아예 출력하지 않음
    → "State the final answer." 같은 메타 코멘트로 끝나는 Gemini의 '사고 과정 로그'가 그대로 들어감
    → 모델이 정답 포맷을 안 내놓도록 학습됨 → 추론 시 fallback(마지막 숫자)이 발동 → 오답
  · 35%가 최종 정답을 본문에서 10회 이상 재진술 (중앙값 7회, 최대 129회)
    → 이게 "앵무새 증후군"의 원인. repetition_penalty=1.15 는 증상 억제일 뿐 원인이 아님
  · \boxed{} 가 있는 샘플은 값 정확도 99.84% → 답 자체는 멀쩡함. 포맷/장황함만 문제
  · gsm8k_real_cot 2,715개는 100% 정상 (짧고, 정확하고, 포맷 완벽) — 이게 골드 스탠다드

출력: deep_chal_math_clean_v4.csv  (약 8,500개, 전부 포맷/정답 검증 통과)
"""
import re
import pandas as pd

SRC = "deep_chal_math_master_train_v3.csv"
LB = "deep_chal_math_leaderboard_filtered.csv"
OUT = "deep_chal_math_clean_v4.csv"

# 학습/추론에서 동일하게 쓸 시스템 프롬프트 (반드시 추론 코드와 일치시킬 것)
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
BOX = "\\boxed{"


def get_solution(formatted_text: str) -> str:
    """formatted_text에서 assistant 응답 부분만 추출"""
    return str(formatted_text).split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()


def last_boxed(text: str):
    """중첩 중괄호 안전하게 마지막 \boxed{...} 내용 추출"""
    idxs = [m.start() for m in re.finditer(re.escape(BOX), text)]
    if not idxs:
        return None
    i = idxs[-1] + len(BOX)
    depth, out = 1, ""
    for ch in text[i:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out += ch
    return out


def as_int_str(x):
    """'1,200' '−3' '5.0' 등을 정규화된 정수 문자열로. 실패 시 None"""
    if x is None:
        return None
    s = str(x).replace(",", "").replace(" ", "").replace("$", "")
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else None


def norm_q(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def build_text(question: str, solution: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{solution}<|im_end|>"
    )


def main():
    df = pd.read_csv(SRC)
    n0 = len(df)
    df["sol"] = df["formatted_text"].apply(get_solution)
    df["ans"] = df["answer"].astype(str).str.strip()
    df["n_box"] = df["sol"].str.count(re.escape(BOX))
    df["box_val"] = df["sol"].apply(last_boxed).apply(as_int_str)
    df["sol_len"] = df["sol"].str.len()
    # 최종 정답 문자열이 풀이에 몇 번 등장하는지 = 재진술 반복 정도
    df["ans_rep"] = [len(re.findall(re.escape(a), s)) for a, s in zip(df["ans"], df["sol"])]

    # 리더보드(평가셋)와 문항이 겹치는 행 표시.
    # 실측: gsm8k_real_cot 255문항이 리더보드와 완전히 동일함.
    # → 학습에서 반드시 제외. (1) 규칙 5.1b 안전, (2) 로컬 검증 점수 부풀림 방지,
    #   (3) 리더보드 점수가 실제 일반화 성능보다 높게 나오는 착시 제거
    lb_keys = set(pd.read_csv(LB)["question"].map(norm_q))
    df["is_eval_overlap"] = df["question"].map(norm_q).isin(lb_keys)

    keep = (
        (df["n_box"] == 1)                       # \boxed{} 정확히 한 번 (마지막에만)
        & (df["box_val"].notna())                # 정수로 파싱 가능
        & (df["box_val"] == df["ans"])           # 그 값이 실제 정답
        & (df["ans_rep"] < 8)                    # 앵무새 제거: 정답 재진술 8회 미만
        & (df["sol_len"] <= 3000)                # 장황한 풀이 제거
        & (~df["is_eval_overlap"])               # 평가셋 유출 제거
    )
    clean = df[keep].copy()

    # 문항 중복 제거 (같은 문제 여러 풀이가 있으면 가장 짧은 풀이 채택 → 간결한 스타일 학습)
    clean = clean.sort_values("sol_len").drop_duplicates(subset=["question"], keep="first")

    # ── 로컬 검증셋 먼저 분리 (리더보드 제출 없이 성능 측정용) ──────────────
    # 편향 없도록 정제 전 '전체' 풀에서 무작위 500문항을 뽑고, 학습셋에서는 제외한다.
    # 평가셋과 분포가 거의 같으므로 신뢰할 만한 프록시.
    # (실측: 리더보드 질문 길이 평균 235자 / train 232자, LaTeX 비율 44% / 42%)
    val = (
        df[~df["is_eval_overlap"]]
        .drop_duplicates(subset=["question"])
        .sample(n=500, random_state=42)
    )
    val[["id", "question", "answer"]].to_csv("local_val_500.csv", index=False)
    val_keys = set(val["question"].map(norm_q))

    clean = clean[~clean["question"].map(norm_q).isin(val_keys)]
    clean["text"] = [build_text(q, s) for q, s in zip(clean["question"], clean["sol"])]
    out = clean[["id", "question", "answer", "source", "text"]].reset_index(drop=True)
    out.to_csv(OUT, index=False)

    print(f"원본 {n0:,} → 정제 {len(out):,} ({len(out)/n0:.1%})")
    print(out["source"].value_counts().to_string())
    print(f"평균 풀이 길이: {int(clean['sol_len'].mean())}자 (원본 {int(df['sol_len'].mean())}자)")
    print(f"저장: {OUT}")
    print(f"로컬 검증셋 저장: local_val_500.csv ({len(val)}문항, 학습 미사용)")


if __name__ == "__main__":
    main()
