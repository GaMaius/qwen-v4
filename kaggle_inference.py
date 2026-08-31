"""
kaggle_inference.py — 제출한 submission.csv 를 그대로 재현한다.

캐글 노트북에서 셀 단위로 실행한 코드를 하나로 합친 것이다.
  · Accelerator : GPU T4 x 2
  · Internet    : Off
  · 소요        : 설치 260초 + 추론 약 4시간 (2,000문항, n=32)

GPU 두 장에 문항을 반씩 나눠 독립 프로세스로 돌린다.
tensor_parallel_size=2 는 /dev/shm 문제로 실패했고, 프로세스를 갈라
CUDA_VISIBLE_DEVICES 로 GPU 를 하나씩 붙이면 NCCL 없이 2배 속도가 나온다.

샤딩은 df.iloc[shard::num_shards] 이므로 단일 GPU 로 돌리면 배치 구성이
달라져 결과가 재현되지 않는다. 반드시 2장으로 나눠 돌릴 것.
"""
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

# 휠 폴더는 탐색해서 찾는다. 데이터셋을 붙이는 방식에 따라
# /kaggle/input/<슬러그>/... 또는 /kaggle/input/datasets/<계정>/<슬러그>/...
# 로 경로가 달라진다.
WHEELS = os.path.dirname(
    sorted(glob.glob("/kaggle/input/**/*.whl", recursive=True))[0])

# ── 1. 오프라인 설치 ────────────────────────────────────────────────
#  --no-index 가 없으면 인터넷 차단 상태에서 PyPI 를 조회하려다 실패한다.
#  torchaudio 는 캐글 기본 이미지의 것이 cu128 이라, vllm 이 가져오는
#  torch(cu130) 와 CUDA 버전이 어긋나 transformers 임포트 단계에서 죽는다.
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                "--find-links", WHEELS, "vllm", "torchvision"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])

# ── 2. 경로 · 사전 점검 ─────────────────────────────────────────────
#  캐글 세션이 재시작되면 pip 설치분이 사라진다.
#  4시간짜리 추론 도중이 아니라 여기서 걸러야 한다.
r = subprocess.run([sys.executable, "-c", "import vllm; print(vllm.__version__)"],
                   capture_output=True, text=True)
assert r.returncode == 0, "vllm 설치 실패:\n" + r.stderr[-500:]
print("vllm", r.stdout.strip())

TEST = glob.glob("/kaggle/input/**/test_submission.csv", recursive=True)
assert len(TEST) == 1, "test_submission.csv 가 %d개: %s" % (len(TEST), TEST)
TEST, = TEST

SC = glob.glob("/kaggle/input/**/02_infer_vllm.py", recursive=True)
assert len(SC) == 1, "02_infer_vllm.py 가 %d개: %s" % (len(SC), SC)
SC, = SC

# 모델 폴더. config.json 첫 매치를 쓰면 노트북에 함께 붙어 있는 "베이스"
# Qwen2.5-3B-Instruct 가 먼저 잡힌다. 이름으로 못 박고 후보를 찍어둔다.
_cands = [os.path.dirname(c)
          for c in glob.glob("/kaggle/input/**/config.json", recursive=True)
          if glob.glob(os.path.dirname(c) + "/*.safetensors")]
print("모델 후보:")
for c in _cands:
    print("   ", c)
MODEL = [c for c in _cands if "qwen_v4_merged" in c]
assert len(MODEL) == 1, (
    "파인튜닝 모델(qwen_v4_merged)을 특정할 수 없습니다. 후보: %s" % _cands)
MODEL, = MODEL

# 크기로 한 번 더 확인. 파인튜닝본은 5.75 GiB 다.
# 파일 복사가 중간에 잘리면 safetensors 헤더 오류로 모델 로딩 단계에서 죽는다.
_gb = sum(os.path.getsize(f)
          for f in glob.glob(os.path.join(MODEL, "*.safetensors"))) / 2 ** 30
assert _gb > 4, "가중치가 %.2f GiB — 파일이 불완전합니다" % _gb

ref = pd.read_csv(TEST)
assert {"id", "question"} <= set(ref.columns), list(ref.columns)
print("입력 %d행 %s\n모델 %s (%.2f GiB)"
      % (len(ref), list(ref.columns), MODEL, _gb))

# 이전 산출물이 남아 있으면 02_infer_vllm.py 가 "이어하기"로 판단해
# 그만큼만 채우고 끝낸다. 노트북을 Commit 으로 재실행할 때 특히 위험하다.
for p in (glob.glob("/kaggle/working/sub_*.csv") + glob.glob("/kaggle/working/bk*")
          + glob.glob("/kaggle/working/submission*.csv")):
    shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)

# ── 3. 추론 ────────────────────────────────────────────────────────
ps = [subprocess.Popen(
        [sys.executable, "-u", SC, "--model", MODEL, "--input", TEST,
         "--output", "/kaggle/working/sub_%d.csv" % i,
         "--shard", str(i), "--num-shards", "2",
         "--n", "32", "--temperature", "0.5", "--top-p", "0.95",
         "--max-tokens", "2048", "--max-model-len", "3072",
         "--chunk", "50", "--backup-dir", "/kaggle/working/bk%d" % i],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)),
        stdout=open("/kaggle/working/log_%d.txt" % i, "w"),
        stderr=subprocess.STDOUT)
      for i in range(2)]
assert [p.wait() for p in ps] == [0, 0], "샤드 실패 — log_0.txt / log_1.txt 확인"

# ── 4. 병합 · 검증 ─────────────────────────────────────────────────
sub = (pd.concat([pd.read_csv("/kaggle/working/sub_%d.csv" % i) for i in range(2)])
         .drop_duplicates("id", keep="last"))
sub = ref[["id"]].merge(sub[["id", "answer"]], on="id")   # 원본 순서 유지

assert len(sub) == len(ref), "%d행 — %d이어야 함" % (len(sub), len(ref))
assert set(sub["id"]) == set(ref["id"]), "id 집합 불일치"
assert sub["answer"].notna().all(), "빈 답 존재"
sub["answer"] = sub["answer"].astype("int64")     # answer 컬럼은 정수만 (규칙 7.2a)
sub.to_csv("/kaggle/working/submission.csv", index=False)

for i in range(2):
    shutil.rmtree("/kaggle/working/bk%d" % i, ignore_errors=True)
    os.remove("/kaggle/working/sub_%d.csv" % i)
print("완료 %d행 | 범위 %d ~ %d"
      % (len(sub), sub["answer"].min(), sub["answer"].max()))
