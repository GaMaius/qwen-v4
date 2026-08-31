"""
05_check_submission.py  —  제출 전 형식 검사 (GPU 불필요, 1초)

8/31 최종 제출은 하루밖에 없다. 그날 형식 문제를 발견하면 손쓸 방법이 없으므로
제출 파일을 만들 때마다 이걸 통과시킨 뒤 올린다.

사용법:
    python 05_check_submission.py --sub submission.csv --ref deep_chal_math_leaderboard_filtered.csv
    python 05_check_submission.py --sub submission.csv --ref test.csv --id-col ID   # 대문자 ID를 요구할 때

--id-col 을 주면 그 이름으로 컬럼을 바꿔서 submission_fixed.csv 로 저장한다.
"""
import argparse
import os
import sys

import pandas as pd

ERRORS = []
WARNS = []


def err(m):
    ERRORS.append(m)


def warn(m):
    WARNS.append(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True, help="검사할 제출 파일")
    ap.add_argument("--ref", required=True, help="정답 id 목록이 있는 원본 (리더보드/테스트 csv)")
    ap.add_argument("--id-col", default=None,
                    help="대회가 요구하는 id 컬럼명 (예: ID). 주면 그 이름으로 고쳐서 저장")
    a = ap.parse_args()

    sub = pd.read_csv(a.sub)
    ref = pd.read_csv(a.ref)
    print(f"제출 {a.sub}: {len(sub):,}행 {list(sub.columns)}")
    print(f"기준 {a.ref}: {len(ref):,}행 {list(ref.columns)}")
    print()

    # ── 컬럼 ────────────────────────────────────────────────
    if len(sub.columns) != 2:
        err(f"컬럼이 2개가 아닙니다: {list(sub.columns)}")
    id_col = sub.columns[0]
    ans_col = sub.columns[1]
    if ans_col != "answer":
        err(f"두 번째 컬럼명이 'answer'가 아닙니다: '{ans_col}'")

    ref_id = "id" if "id" in ref.columns else ref.columns[0]

    # ── 행 수와 id 일치 ──────────────────────────────────────
    s_ids, r_ids = set(sub[id_col].astype(str)), set(ref[ref_id].astype(str))
    missing, extra = r_ids - s_ids, s_ids - r_ids
    if missing:
        err(f"빠진 id {len(missing)}개 (빈 값은 오답 처리됨). 예: {sorted(missing)[:5]}")
    if extra:
        err(f"기준에 없는 id {len(extra)}개. 예: {sorted(extra)[:5]}")
    if sub[id_col].duplicated().any():
        n = int(sub[id_col].duplicated().sum())
        err(f"중복 id {n}개. 예: {sub[id_col][sub[id_col].duplicated()].head(3).tolist()}")

    # ── answer 값 ───────────────────────────────────────────
    if sub[ans_col].isna().any():
        err(f"빈 answer {int(sub[ans_col].isna().sum())}개")
    nums = pd.to_numeric(sub[ans_col], errors="coerce")
    if nums.isna().any() and not sub[ans_col].isna().any():
        bad = sub[nums.isna()][ans_col].head(3).tolist()
        err(f"숫자가 아닌 answer {int(nums.isna().sum())}개. 예: {bad}")
    ok_nums = nums.dropna()
    if len(ok_nums):
        non_int = (ok_nums != ok_nums.round()).sum()
        if non_int:
            err(f"정수가 아닌 answer {int(non_int)}개 (대회는 정수만 허용)")
        if (ok_nums.abs() > 10 ** 15).any():
            err(f"비정상적으로 큰 값 {int((ok_nums.abs()>10**15).sum())}개 — 환각 의심")
        z = int((ok_nums == 0).sum())
        if z > len(ok_nums) * 0.05:
            warn(f"answer가 0인 행이 {z}개 ({z/len(ok_nums):.1%}) — 추출 실패가 많을 수 있음")

    # ── 결과 ────────────────────────────────────────────────
    for m in WARNS:
        print(f"  [경고] {m}")
    if ERRORS:
        print()
        for m in ERRORS:
            print(f"  [오류] {m}")
        print("\n❌ 이대로 제출하면 안 됩니다.")
        sys.exit(1)

    print("✅ 형식 검사 통과")
    print(f"   {len(sub):,}행, id 완전 일치, answer 전부 정수")

    if a.id_col and id_col != a.id_col:
        out = os.path.join(os.path.dirname(os.path.abspath(a.sub)) or ".",
                           "submission_fixed.csv")
        sub2 = sub.rename(columns={id_col: a.id_col})
        sub2[a.id_col] = sub2[a.id_col].astype(str)
        sub2.to_csv(out, index=False)
        print(f"\n   id 컬럼 '{id_col}' → '{a.id_col}' 로 바꿔 저장: {out}")


if __name__ == "__main__":
    main()
