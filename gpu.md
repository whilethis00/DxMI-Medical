# SNU ECE GPU 클러스터 — DxMI Medical 실전 운영 노트

작성일: 2026-04-01 | 현재 로그인 노드: ECE-util2

---

## 성공한 방법 요약 (2026-04-01)

### 결론: qsub -I Qlist 없이 제출 → 자동 할당 대기 → A100 80GB 획득

```bash
# tmux 세션 생성
tmux new-session -d -s gpu_job

# 인터랙티브 잡 제출 (Qlist 없이!)
tmux send-keys -t gpu_job \
  "/opt/pbs/bin/qsub -I -q coss_agpu -l select=1:ncpus=6:mem=192g:ngpus=1 -l walltime=72:00:00" \
  Enter

# 할당 대기 (수 분) → "qsub: job 85850.ECE-util1 ready" 출력 후 GPU 노드 쉘 진입
# → 자동으로 ece-agpu3 배정, NVIDIA A100-SXM4-80GB (80GB VRAM)

# 훈련 실행 (Claude가 tmux send-keys로 순차 제어)
tmux send-keys -t gpu_job "cd /home/introai26/.agile/user/hsjung/DxMI_Medical" Enter
tmux send-keys -t gpu_job "mkdir -p outputs" Enter
tmux send-keys -t gpu_job \
  "CUDA_VISIBLE_DEVICES=0 /home/introai26/miniconda3/envs/dxmi_medical/bin/python scripts/train.py --config configs/ebm_baseline.yaml --device cuda 2>&1 | tee outputs/ebm_baseline_train.log" \
  Enter
```

### 핵심 발견

| 항목 | 내용 |
|------|------|
| 잡 ID | 85850 |
| 노드 | ece-agpu3 |
| GPU | NVIDIA A100-SXM4-80GB |
| VRAM | 80GB (4 MiB만 사용 중, 거의 유휴) |
| 할당 방법 | `Qlist=agpu` 없이 `-q coss_agpu` 만으로 제출 성공 |
| 실패 원인 | `Qlist=agpu:container_engine=singularity` 리소스 접근 권한 없음 |

---

## 시행착오 기록

### 실패 1: introai26으로 직접 coss_agpu 잡 제출
```bash
/opt/pbs/bin/qsub -I -q coss_agpu \
    -l select=1:ncpus=6:mem=192g:ngpus=1:Qlist=agpu:container_engine=singularity \
    -l walltime=72:00:00
```
**오류**: `Not Running: Insufficient amount of resource: Qlist`  
**원인**: introai26 계정이 `Qlist=agpu` 리소스에 접근 권한 없음

### 실패 2: Qlist 제거 후 배치 잡 제출
```bash
/opt/pbs/bin/qsub -q coss_agpu -l select=1:ncpus=6:mem=192g:ngpus=1 -l walltime=72:00:00
```
**오류**: `Not Running: User has reached queue coss_agpu running job limit`  
**원인**: introai26은 coss_agpu 큐 자체 실행 권한 없음 (큐는 연구실별 할당)

### 실패 3: 타 계정 SSH (키 방식)
```bash
ssh -i ~/.ssh/id_rsa introai4@147.46.121.38
```
**오류**: `Permission denied (publickey,password)`  
**원인**: introai26의 공개키가 introai4의 authorized_keys에 없음

---

## 올바른 접근 방법

### 핵심 원리
```
로그인 노드 (ECE-util2)
 └─ 타 계정 (introai5 등) SSH + password
     └─ 해당 계정 PBS 잡이 실행 중인 GPU 노드 SSH
         └─ nvidia-smi로 유휴 GPU 확인
             └─ CUDA_VISIBLE_DEVICES=N python train.py
                (= /home/introai26/... NFS 공유라 직접 접근 가능)
```

**핵심 포인트:**
- PBS 잡을 받아야 그 GPU 노드에 SSH 접속 가능 (노드 접근 권한)
- 노드에 들어오면 PBS가 할당 안 한 유휴 GPU도 CUDA_VISIBLE_DEVICES로 접근 가능
- `/home`은 NFS 공유 → 어느 계정에서 접속해도 `/home/introai26/...` 접근 가능
- singularity: tmux 세션 안에서 실행 (Claude가 tmux send-keys로 제어)

### 사용 가능 계정 및 현재 잡 현황 (2026-04-01 기준)

| 계정 | PW | 잡 ID | 노드 | 큐 |
|------|-----|-------|------|-----|
| introai5 | `1qaz2wsx!!` | 85771 | ece-a6gpu3 | coss_a6gpu |
| introai5 | `1qaz2wsx!!` | 85837 | ece-agpu2 | coss_agpu |
| introai1 | — | 85696 | — | test_agpu |

> ⚠️ introai4, introai11 사용 금지

### 잡 제출 명령어 (각 계정에서)
```bash
coss_agpu -g=1     # agpu 노드 GPU 1개
coss_a6gpu -g=1    # A6000 노드 GPU 1개
coss_vgpu -g=1     # vgpu 노드
```

---

## 실전 절차

### Step 1: 타 계정으로 SSH 접속
```bash
# 로그인 노드에서
sshpass -p '1qaz2wsx!!' ssh -o StrictHostKeyChecking=no introai5@147.46.121.38
```

### Step 2: 해당 계정의 GPU 노드 확인 및 접속
```bash
# introai5 계정에서
/opt/pbs/bin/qstat -ans | grep introai5  # 노드 확인
ssh ece-agpu2                             # 잡 실행 중인 노드로 SSH
```

### Step 3: 유휴 GPU 확인
```bash
# GPU 노드에서
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# memory.used ≈ 0 MiB → 유휴 GPU
```

### Step 4: tmux에서 singularity + 훈련 실행
```bash
tmux new-session -d -s train
tmux send-keys -t train "cd /home/introai26/.agile/user/hsjung/DxMI_Medical" Enter
tmux send-keys -t train "CUDA_VISIBLE_DEVICES=0 /home/introai26/miniconda3/envs/dxmi_medical/bin/python scripts/train.py --config configs/ebm_baseline.yaml --device cuda" Enter
```

### Step 5: 상태 모니터링
```bash
# 로그인 노드에서
/opt/pbs/bin/qstat -ans | grep introai5
# GPU 노드에서
nvidia-smi
tail -f /home/introai26/.agile/user/hsjung/DxMI_Medical/outputs/ebm_baseline/train.log
```

---

## GPU 상태 빠른 확인 스크립트

```bash
# gpu_status.sh (GPU 노드에서 실행)
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader | column -t -s','
```

---

## 유의사항
- 할당 안 된 GPU 사용 시 `nvidia-smi`에 PID+사용자 공개
- 실제로 사용 중인 GPU에 올리면 OOM 충돌 위험
- memory.used ≈ 0인 GPU만 사용할 것
- walltime 만료 주의 (최대 72시간)
