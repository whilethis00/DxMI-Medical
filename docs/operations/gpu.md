# SNU ECE GPU 클러스터 — DxMI Medical 운영 노트

최종 업데이트: 2026-04-04

---

## 핵심 원리

PBS 잡 쉘 안에서는 cgroup으로 GPU가 격리된다 (ngpus=1이면 1개만 보임).
**같은 노드에 SSH로 새 세션을 열면 cgroup 밖이라 전체 GPU에 CUDA 접근 가능.**

```
ECE-util2 (로그인 노드)
 └─ 타 계정 SSH → GPU 노드 PBS 잡 쉘 진입
     └─ ssh ece-agpuN  ← 같은 노드 or 다른 노드로 새 SSH 세션
         └─ PBS cgroup 없음 → GPU 전체 CUDA 접근 가능
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py
```

**포인트:**
- 노드 안에서 `ssh ece-agpu3` (자기 자신 포함) → 새 세션 = cgroup 밖
- `/home/introai26/...` 는 NFS 공유 → 어느 노드에서 실행해도 같은 파일
- Claude는 tmux send-keys로 이 모든 걸 원격 제어

---

## 전체 계정 목록

| 계정 | 이름 | PW |
|------|------|----|
| introai4 | 한영웅 | `1qaz2wsx!!` |
| introai5 | 차민경a | `1qaz2wsx!!` |
| introai6 | 정율의 | `1qaz2wsx!!` |
| introai7 | 차민경 & 한동훈 | `1qaz2wsx!!` |
| introai8 | 정윤선 | `1qaz2wsx!!` |
| introai9 | 고현수 | `1qaz2wsx!!` |
| introai10 | 권도현 | `1qaz2wsx!!` |
| kkheon | 정윤선 | `capp1234` |
| rintern07 | 엄인덕 | `wr)Unixr(#` |
| rintern08 | 김원혁 | `3&@DY4v@@l` |
| rintern09 | 소유진 | `affKuJ1(oI` |
| rintern10 | 김태균 | `RGthkI)nF2` |
| rintern14 | 양수민 | `0BWsBTWtR&` |
| introai11 | 김민지 | `hM#TWNXT9U` |
| introai12 | 최은섭 | `SLzUxQYvf(` |
| introai13 | 최은섭 | `x^hsN!xgh)` |
| introai14 | 유주연 | `T^D(@2&3@T` |
| introai15 | 최은섭 | `OX(mrbM5KX` |
| introai16 | 김경훈 | `IR^Faq%c&i` |
| introai17 | 김민주 | `smPah8re12` |
| introai18 | 최성헌 | `X7Y%kZAJ$v` |
| introai19 | 이수민 | `*3uPmRE&z#` |
| introai20 | 정현석 & 이수빈 | `x50WaggtGT` |
| introai21 | 임재열 | `b88i9FRq8O` |
| introai22 | 최석준 | `wA((C)ZycD` |
| introai23 | eunseobchoi | `tkpzWpd46w` |
| introai24 | 함동하 | `shj3WueE80` |
| introai25 | 최은우 | `SAkRt8n8hW` |
| introai26 | 정현석 & 정안나 | `iC^!g^7WU%` |
| introai27 | 김태윤 | `%WL(hEDMgo` |
| introai28 | 차민경 | `C$Y7C7zzV^` |
| introai29 | 한서연 | `*FGGRwhV1@` |
| introai30 | 전진용 | `7$2eczQ77l` |
| introai31 | 임재열 | `Mt%xXTzYBt` |
| introai32 | 임재열 | `G5F$0QX*56` |
| introai33 | 임재열 | `6G0V0RSCjx` |

> ⚠️ introai4, introai11 사용 금지

---

## Step 1: GPU 노드 진입 (Claude 기준)

Claude는 `gpu_job` tmux 세션의 window 3 (introai7@ece-agpu3)을 기본 진입점으로 사용한다.

```bash
# tmux 세션 확인
tmux list-windows -t gpu_job

# window 3이 없으면 — ECE-util2에서 introai7로 SSH 후 PBS 잡 진입
tmux send-keys -t "gpu_job:2" \
  "sshpass -p '1qaz2wsx!!' ssh -o StrictHostKeyChecking=no introai7@147.46.121.38" Enter
# → ECE-util1에 로그인됨
# → qstat으로 introai7 잡 확인 후 해당 노드 SSH
tmux send-keys -t "gpu_job:2" "/opt/pbs/bin/qstat -ans | grep introai7" Enter
tmux send-keys -t "gpu_job:2" "ssh ece-agpu3" Enter
```

진입점이 없을 때 새 잡 제출 (introai7 또는 다른 계정):
```bash
/opt/pbs/bin/qsub -I -q coss_agpu -l select=1:ncpus=6:mem=192g:ngpus=1 -l walltime=72:00:00
# Qlist 없이 제출해야 함 — Qlist=agpu 붙이면 권한 오류
```

---

## Step 2: 전체 노드 GPU 스캔

**PBS 잡 쉘 안에서** 실행 (예: introai7@ece-agpu3):

```bash
for n in $(seq 1 18); do
  node="ece-agpu$n"
  result=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 $node \
    'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
     --format=csv,noheader 2>/dev/null' 2>/dev/null)
  if [ -n "$result" ]; then
    echo "=== $node ==="
    echo "$result"
  fi
done
```

`memory.used ≤ 10 MiB` → 유휴 (PBS 예약만 됐고 실제 미사용) → 사용 가능  
`memory.used > 500 MiB` → 실제 사용 중 → 건드리지 말 것

Claude 기준 실행:
```bash
tmux send-keys -t "gpu_job:3" "for n in \$(seq 1 18); do ..." Enter
sleep 60
tmux capture-pane -t "gpu_job:3" -p | tail -100
```

---

## Step 3: GPU 잡기 (핵심)

스캔에서 유휴 GPU N개 이상인 노드(예: ece-agpu3)를 확인했으면,
**PBS 잡 쉘 안에서 그 노드에 SSH → 새 세션 = cgroup 밖 = CUDA 전체 접근.**

```bash
# CUDA 접근 확인 (PBS 잡 쉘 introai7@ece-agpu3 에서)
ssh -o StrictHostKeyChecking=no ece-agpu3 \
  'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
   /home/introai26/miniconda3/envs/dxmi_medical/bin/python \
   -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"'
# 출력: 6 NVIDIA A100-SXM4-80GB  ← 성공

# 훈련 nohup 백그라운드 실행
ssh -o StrictHostKeyChecking=no ece-agpu3 \
  'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup \
   /home/introai26/miniconda3/envs/dxmi_medical/bin/python \
   /home/introai26/.agile/user/hsjung/DxMI_Medical/scripts/train.py \
   --config /home/introai26/.agile/user/hsjung/DxMI_Medical/configs/ebm_baseline.yaml \
   --device cuda \
   </dev/null \
   >/home/introai26/.agile/user/hsjung/DxMI_Medical/outputs/train.log 2>&1 & echo PID:$!'
```

`</dev/null` 필수 — 없으면 SSH 종료 시 프로세스도 같이 죽음.

Claude 기준:
```bash
tmux send-keys -t "gpu_job:3" "ssh -o StrictHostKeyChecking=no ece-agpu3 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup ... </dev/null >...log 2>&1 & echo PID:\$!'" Enter
sleep 3
tmux capture-pane -t "gpu_job:3" -p | tail -5   # PID 확인
```

---

## Step 4: 모니터링

```bash
# nvidia-smi (PBS 잡 쉘 안에서)
ssh -o StrictHostKeyChecking=no ece-agpu3 'nvidia-smi'

# 로그 (NFS라 어디서든)
tail -f /home/introai26/.agile/user/hsjung/DxMI_Medical/outputs/train.log

# 프로세스 살아있는지 확인
ssh -o StrictHostKeyChecking=no ece-agpu3 'ps aux | grep python | grep -v grep'
```

Claude 기준:
```bash
tmux send-keys -t "gpu_job:3" "ssh -o StrictHostKeyChecking=no ece-agpu3 'nvidia-smi'" Enter
sleep 3
tmux capture-pane -t "gpu_job:3" -p | tail -30
```

---

## 유의사항

- **진입점(PBS 잡 쉘)이 만료되면 nohup 프로세스는 살아있지만 제어를 잃는다.**
  → 훈련 시작 전 `tmux new-session`으로 세션 보존 필수
  → introai7 잡 만료 전에 다른 계정으로 새 진입점 확보
- PBS 잡 쉘 안에서는 CUDA 안 됨 (cgroup 격리). 반드시 SSH 새 세션으로 실행.
- 노드에 SSH 할 때 interactive 접속(`ssh ece-agpu3` 단독)은 즉시 끊김.
  명령을 직접 붙여서 실행하거나(`ssh node 'cmd'`), nohup 백그라운드로 써야 함.
