# ebm_fm_gate_v1 — Result

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-21 |
| **베이스** | ebm_weighted_cd (C, 3-seed avg: ρ=0.239±0.010, AUROC=0.688 on test) |
| **목적** | FM gate로 SGLD warm-up 후 FM hybrid negative 투입 — trivial negative 문제 해소 + ranking 성능 향상 |
| **설정** | epochs=30, l2_reg=0.1, reward_cd_weight=1.0, fm_gate_sep_std_threshold=40→버그수정→재실행, fm_gate_warmup_steps=570, sgld_permanent_ratio=0.2, fm_gate_probe_size=16, seed=1 |
| **현황** | 학습 완료 (ep30) / val 완료 / **test set 평가 미완료** |

---

## 2. 무엇을 검증하나

**맥락**: 원래 C(MaxEnt IRL)는 FM 샘플을 negative로 쓰다가 EBM이 real vs noise를 trivially 이진 분리하는 collapse(sep/std ≈ 130)가 발생했음. ebm_weighted_cd(C r1~r3)는 이 문제를 우회하기 위해 **의도적으로 SGLD-only**로 돌려 "EBM + reward-weighted CD" 핵심 메커니즘이 작동하는지 먼저 검증한 것. 그 결과 A < B < C (test ρ=0.239) 성립을 확인.

**이번 실험의 질문**: SGLD-only로 검증이 끝났으니, 이제 FM을 다시 투입해도 collapse 없이 성능을 더 올릴 수 있는가?

**접근**:
1. Gated hybrid negative: EBM이 충분히 학습된 후(sep/std EMA < threshold 연속 3회)에야 FM negative 투입
2. SGLD 20%는 전 기간 유지해 hard negative baseline 보존
3. FM gate 버그 수정 (`fm_gate_warmup_steps=570` 추가 — 초기 무작위 EBM에서 gate가 오작동하는 문제)

**합격 기준**: val Spearman ρ > 0.20 (p < 0.05), AUROC(E) > 0.65 — C best(0.239)에 근접하거나 초과

---

## 3. 학습 손실 곡선

### 주요 epoch 손실 요약 (second run 기준, 본 실험)

| epoch | step | rw | cd | reg | e+(std) | e-(std) | ∇rw | fm |
|-------|------|----|----|-----|---------|---------|-----|----|
| 1 | s0010 | -0.46 | -0.55 | 0.09 | -0.84(±0.06) | -0.30(±0.00) | 16.4 | off |
| 1 | s0190 | -4.94 | -9.77 | 4.83 | -4.78(±0.74) | +4.96(±0.01) | 4.0 | off |
| 2 | s0220 | +2.83 | -1.17 | 4.00 | -3.80(±0.15) | -2.64(±4.26) | 65.3 | **ON** ← gate 개방 |
| 5 | — | — | — | — | — | — | — | ON |
| 10 | s1900 | ≈-5.0 | ≈-10.0 | ≈5.0 | ≈-5.0(±0.3) | ≈+5.0(±0.0) | 2~10 | ON |
| 28 | s5320 | -4.99 | -10.05 | 5.06 | -4.95(±0.35) | +5.09(±0.09) | 2.3 | ON |
| 30 | s5700 | -4.86 | -9.43 | 4.57 | -4.58(±0.58) | +4.93(±0.04) | 7.8 | ON |

**관찰**:
- ep1 초반: e+ ≈ e- ≈ -0.3 (EBM 미학습 상태) → s0190에서 e+≈-5, e-≈+5로 빠르게 분리
- FM gate: ep02/s0220 (reward step ≈1100 > warmup 570)에서 개방. sep_std_ema가 threshold 조건 통과
- gate 개방 직후: ∇rw spike (65~372) 다수 발생 — FM 샘플이 SGLD와 혼합되면서 EBM에 harder signal
- ep10 이후: 손실 안정화, ∇rw 2~15 수준으로 정착. 간헐적 spike(50~600) 산발적으로 지속
- **sep=32.7 고착**: gate 개방 후 sep_std_ema 업데이트 중단 (gate 체크 로직이 disabled되기 때문) — 로깅 공백

### 첫 번째 실행 조기 종료 (버그 확인)

| step | 현상 | 판단 |
|------|------|------|
| ep01/s0040 | FM gate 오작동 개방 (sep_std=9.3, warmup 없음) | EBM 미학습 상태에서 sep_std 우연히 threshold 통과 |
| ep01/s0160 | ∇rw=486.54 (gradient spike) | FM noise 샘플이 대량 투입되며 불안정 |
| ep01/s0180 | e+=-0.41(±5.25) | demo energy 분산 폭발, 학습 붕괴 징후 |
| ep01 val | ρ=+0.052 FAIL | 조기 종료 판단, warmup 추가 후 재실행 |

---

## 4. 검증 지표

### Epoch별 val 결과

| epoch | Spearman ρ | p-value | AUROC(E) | ECE | 판정 | 비고 |
|-------|-----------|---------|----------|-----|------|------|
| 1 | -0.1223 | 0.1050 | 0.5015 | 0.7707 | FAIL | |
| 2 | +0.0040 | 0.9580 | 0.4696 | 0.6983 | FAIL | |
| 3 | +0.0209 | 0.7823 | 0.5116 | 0.5563 | FAIL | |
| 4 | +0.0141 | 0.8518 | 0.4985 | 0.7505 | FAIL | |
| 5 | +0.1355 | 0.0722 | 0.6107 | 0.7946 | FAIL | |
| 6 | +0.0970 | 0.1991 | 0.5561 | 0.7867 | FAIL | |
| 7 | -0.0460 | 0.5431 | 0.4779 | 0.7029 | FAIL | |
| 8 | +0.0888 | 0.2397 | 0.5180 | 0.7859 | FAIL | |
| 9 | +0.0131 | 0.8622 | 0.6087 | 0.7841 | FAIL | |
| 10 | +0.1390 | 0.0650 | 0.6189 | 0.7948 | FAIL | |
| 11 | +0.1200 | 0.1117 | 0.5545 | 0.7938 | FAIL | |
| 12 | +0.1742 | 0.0204 | 0.6044 | 0.7945 | **PASS** | 첫 PASS |
| 13 | +0.0780 | 0.3021 | 0.5319 | 0.7821 | FAIL | |
| 14 | +0.1481 | 0.0491 | 0.5641 | 0.7860 | PASS | |
| 15 | +0.0788 | 0.2974 | 0.5423 | 0.7273 | FAIL | |
| 16 | +0.1680 | 0.0254 | 0.5887 | 0.7935 | PASS | |
| 17 | +0.1884 | 0.0120 | 0.6845 | 0.7929 | PASS | |
| 18 | +0.2246 | 0.0026 | 0.6525 | 0.7944 | PASS | |
| 19 | +0.2609 | 0.0005 | 0.6676 | 0.7954 | PASS | |
| 20 | +0.0059 | 0.9379 | 0.5188 | 0.7963 | FAIL | 이상 하락 |
| 21 | +0.2187 | 0.0035 | 0.6484 | 0.7964 | PASS | |
| 22 | +0.1975 | 0.0084 | 0.7111 | 0.7946 | PASS | |
| 23 | +0.1875 | 0.0125 | 0.5986 | 0.7936 | PASS | |
| 24 | +0.1995 | 0.0078 | 0.5967 | 0.7820 | PASS | |
| 25 | +0.2114 | 0.0047 | 0.6319 | 0.7888 | PASS | |
| 26 | +0.1500 | 0.0463 | 0.5370 | 0.7909 | PASS | |
| 27 | +0.1623 | 0.0309 | 0.5749 | 0.7872 | PASS | |
| **28** | **+0.2791** | **0.0002** | **0.6693** | 0.7945 | **PASS** | **BEST val** |
| 29 | +0.2604 | 0.0005 | 0.6710 | 0.7900 | PASS | |
| 30 | +0.2115 | 0.0047 | 0.6657 | 0.7936 | PASS | |

### 이전 실험 대비 비교 (val 기준)

| 실험 | best val ρ | best epoch | AUROC(E) | ECE | 비고 |
|------|-----------|-----------|----------|-----|------|
| A (no IRL) | -0.0878 | 10 | 0.3884 | 0.7956 | FAIL |
| B (supervised) | +0.3028 | 80 | 0.6719 | 0.2095 | PASS |
| C r1 (SGLD-only) | +0.2364 | 14 | 0.6653 | 0.7933 | PASS |
| C r2 | +0.2902 | 17 | 0.6767 | 0.7936 | PASS |
| C r3 | +0.2946 | 20 | 0.7074 | 0.7934 | PASS |
| **FM gate v1** | **+0.2791** | **28** | **0.6693** | 0.7945 | PASS |

- FM gate v1 best val ρ(0.2791)은 C r1(0.2364)을 상회하고 C r2(0.2902)에 근접 ✅
- 단, C r3 best(0.2946)에는 미치지 못함. **seed 1 단일 run 결과**이므로 직접 비교는 유보
- **test set 평가 필수** — val 수치만으로 우위 선언 불가

---

## 5. 결과 해석 및 인사이트

### 긍정적

- **FM gate 메커니즘 작동 확인**: warmup(570 steps) + consecutive check(3회) 조합이 early gate 오작동을 방지. ep02/s0220에서 정상적으로 개방, 이후 학습 안정화
- **val ρ 상승 궤적**: ep12 첫 PASS 이후 ep19(ρ=0.261), ep28(ρ=0.279)로 후반부에도 지속 개선. SGLD-only C와 달리 ep20 이상에서 plateau 없이 성장
- **ep20 이상 FAIL 없음** (ep20 제외): C r1은 ep8~20 사이 3회 FAIL했지만 FM gate v1은 ep20 이후 전 epoch PASS — FM이 hard negative를 꾸준히 공급하는 효과로 추정
- **AUROC 후반 안정**: ep22(0.711), ep29(0.671), ep30(0.666) — SGLD-only C와 유사 수준 유지

### 주의

- **∇rw 간헐적 spike**: ep1~5에서 65~624, ep10 이후에도 산발적으로 45~100 수준 spike. FM 샘플이 e-를 불안정하게 만드는 것으로 보임. grad_clip=1.0이 일부 흡수하지만 source는 미해결
- **ep20 이상 하락**: ρ=0.261(ep19) → 0.006(ep20). 단발성 spike로 보이지만 원인 불명. e_neg_std가 해당 epoch에서 과도하게 커질 때 발생하는 패턴과 일치
- **ECE ≈ 0.793 고착**: SGLD-only C와 동일한 수준. FM hybrid가 calibration에 기여하지 못함. B의 ECE=0.232와 격차 여전히 큼
- **fm_e=+nan 로깅 공백**: gate 개방 이후 FM 샘플 에너지 모니터링이 중단됨 → **당일 코드 수정 완료** (`x_neg[n_sgld:]`의 에너지를 로깅하도록 수정)
- **test set 평가 없음**: val 결과만으로 C 대비 우위 주장 불가

### 근본 원인 가설

FM gate 개방 후 `sep_std_ema` 업데이트가 완전히 중단되어, FM negative의 품질을 실시간으로 모니터링하지 못함. 실제로 sep=32.7이 ep02/s0220부터 ep30/s5700까지 변함없이 고착 — EBM이 FM 샘플을 얼마나 잘 분리하는지 알 수 없는 상태에서 30 epoch 전체를 운용함.

---

## 6. 다음 스텝

- [ ] **test set 평가**: `eval_test.py --config configs/ebm_fm_gate_v1.yaml` — ckpt_epoch0028.pt 기준 (best val)
- [ ] **seed 2, 3 재현**: FM gate v1 3-seed 평균으로 C 3-seed 평균과 공정 비교 (val이 아닌 test 기준)
- [ ] **fm_gate_v2 설계**: sep_std_ema를 gate 개방 후에도 계속 업데이트 (현재는 중단됨) — FM 품질 실시간 모니터링 + adaptive gate
- [ ] **∇rw spike 원인 분석**: FM 샘플이 포함된 batch에서 e_neg_std가 폭발하는 step 조건 분석. `e_neg_std > threshold`일 때 해당 FM 샘플을 drop하거나 clamp 강화하는 방어 로직 검토
- [ ] **ECE 개선**: post-hoc temperature scaling — B(ECE=0.232) vs C/FM gate(ECE≈0.79) 격차 논문 대응 필요

---

## 7. 저장 파일 목록

```
outputs/ebm_fm_gate_v1_20260421/
├── train.log           # 전체 학습 로그 (2-run: 버그 발견 run + 본 run)
├── RESULT.md           # 이 파일
├── ckpt_epoch0028.pt   # best val checkpoint (ρ=0.2791) — git 제외
└── ckpt_epoch0030.pt   # last checkpoint — git 제외
```
