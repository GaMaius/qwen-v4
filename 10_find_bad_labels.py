"""
10_find_bad_labels.py — 라벨 불량 문항 목록 작성

두 출처를 합친다.

(1) 커뮤니티 제보 목록
    주최측이 "의도된 것이 아니라 원본 소스 단계에서 유입된 오류"로 인정했고,
    "참가자분들께서 각자 판단하여 학습에서 제외"하라고 안내한 목록.
    (2026-08-18 운영팀 답변)

(2) 자동 탐지 — 모델 합의를 신호로 쓴다
    "합의율이 높은데 정답이 0개" = 모델이 확신을 갖고 일관되게 X를 답했는데 라벨은 Y.
    모델이 못 푼 게 아니라 라벨이 틀렸을 가능성이 높다.

    이 판별식은 제보 목록으로 검증했다 (18,319문항 실측):
        · 제보 mislabel 문항의 66.3%가 0정답 (나머지 문항은 15.8%)
        · 0정답 문항 중 제보 오류일 확률: 합의율 <0.3 에서 14.1%,
          0.5~0.7 에서 35.1%, ≥0.7 에서 48.6%
        · 제보 mislabel 442개 중 64.3%에서 모델 최빈답 = 제보자가 손으로 계산한 값
          (독립적인 두 출처가 같은 답에 도달)

왜 제외하는가
    라벨이 틀린 문항에서는 거절 샘플링이 거꾸로 작동한다.
    모델이 맞게 풀면 정답 대조에 실패해 버려지고,
    틀린 답이 우연히 틀린 라벨과 일치하면 그 풀이가 학습 데이터로 들어간다.

사용법
  python 10_find_bad_labels.py --agree-threshold 0.7 --out bad_label_ids.csv
"""
import argparse
import glob
import os
import re

import pandas as pd

REPORTS = [
    "C:/Users/rkfka/Downloads/organizer_report_mislabel_442.csv",
    "C:/Users/rkfka/Downloads/organizer_report_illposed_623.csv",
]
EXTRA_TEXT = "C:/Users/rkfka/Downloads/message.txt"   # 자유 형식 제보. train-###### 을 긁는다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-dir", default="./rft_out")
    ap.add_argument("--pool", default="rft_pool.csv")
    ap.add_argument("--agree-threshold", type=float, default=0.7,
                    help="0정답 문항 중 이 합의율 이상이면 라벨 불량 의심으로 추가")
    ap.add_argument("--out", default="bad_label_ids.csv")
    args = ap.parse_args()

    reported = set()
    for f in REPORTS:
        if os.path.isfile(f):
            reported |= set(pd.read_csv(f)["id"].astype(str))
    if os.path.isfile(EXTRA_TEXT):
        reported |= set(re.findall(r"train-\d{6}",
                                   open(EXTRA_TEXT, encoding="utf-8").read()))
    print(f"제보 목록: {len(reported):,}개")

    dbg = pd.concat([pd.read_csv(f) for f in
                     glob.glob(os.path.join(args.debug_dir, "*_debug.csv"))],
                    ignore_index=True).drop_duplicates("id", keep="last")
    pool = pd.read_csv(args.pool)
    d = dbg.merge(pool, on="id", suffixes=("_pred", "_gt"))

    # 자동 탐지: 정답이 하나도 안 나왔는데 모델은 한 답으로 강하게 수렴한 문항
    sus = d[(d["n_correct"] == 0) & (d["agree"] >= args.agree_threshold)]
    auto = set(sus["id"].astype(str))
    new = auto - reported
    print(f"자동 탐지(0정답 & 합의율≥{args.agree_threshold}): {len(auto):,}개 "
          f"— 그중 제보에 없던 것 {len(new):,}개")

    bad = reported | auto
    in_pool = pool[pool["id"].astype(str).isin(bad)]

    out = in_pool[["id", "question", "answer"]].copy()
    out["source"] = ["제보" if str(i) in reported else "자동탐지"
                     for i in out["id"]]
    # 모델이 대신 내놓은 답. 제보자의 suggested_answer 와 대조하거나 눈으로 볼 때 쓴다.
    out = out.merge(d[["id", "answer_pred", "agree", "n_correct"]], on="id", how="left")
    out.to_csv(args.out, index=False)

    print(f"\n총 {len(bad):,}개 중 rft_pool 에 있는 것 {len(out):,}개 → {args.out}")
    print(out["source"].value_counts().to_string())

    # 이 문항들이 "모델이 못 푸는 문제"로 잘못 집계되고 있었는지 확인.
    zero = int((d["n_correct"] == 0).sum())
    zero_bad = int(((d["n_correct"] == 0) & d["id"].astype(str).isin(bad)).sum())
    print(f"\n0정답 문항 {zero:,}개 중 라벨 불량으로 설명되는 것: {zero_bad:,}개 "
          f"({zero_bad/max(zero,1):.1%})")
    print(f"→ 실제로 모델이 못 푸는 문항은 {zero - zero_bad:,}개 "
          f"({(zero-zero_bad)/len(d):.1%})")
    print(f"→ 보정 pass@8 = {1-(zero-zero_bad)/len(d):.3f} "
          f"(보정 전 {1-zero/len(d):.3f})")


if __name__ == "__main__":
    main()
