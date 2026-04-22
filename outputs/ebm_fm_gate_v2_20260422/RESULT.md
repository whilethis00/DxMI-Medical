# ebm_fm_gate_v2 — Result

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-22 |
| **베이스** | ebm_fm_gate_v1 (best val ρ=0.2791 @ ep28, seed=1) |
| **목적** | v1 대비 energy_clamp=20.0 추가 — ep20 collapse 및 ∇rw spike 억제 효과 측정 |
| **설정** | v1과 완전 동일, energy_clamp: null → 20.0 만 변경. seed=1 고정 |
| **현황** | 학습 중 (2026-04-22 15:35 시작, 예상 완료 21~22시) |

---

## 2. 무엇을 검증하나

### 이전 실험(v1)에서 발견된 문제

v1은 30 epoch을 완주했고 best val ρ=0.2791을 기록했지만, 두 가지 불안정 요소가 해소되지 않았다.

**문제 1 — ep20 collapse**: ep19에서 ρ=0.261로 올라갔다가 ep20에서 ρ=0.006으로 한 epoch 만에 붕괴했다. 로그를 보면 해당 epoch 직전에 e_neg_std가 4.26까지 폭발한 step이 있었다. 쉽게 말하면 FM 샘플 중 일부가 real nodule과 거의 같은 에너지를 가지게 되면서 CD loss가 뒤집히는 현상이다. 이번엔 한 epoch 만에 회복됐지만, 조건이 나쁘면 회복 못하고 그대로 망할 수 있다.

**문제 2 — ∇rw 간헐적 spike**: ep1~5 구간에서 최대 624, 이후에도 40~100 수준의 spike가 계속 발생했다. grad_clip=1.0이 일부를 흡수하지만, clip 되기 전 gradient가 이 크기면 EBM 파라미터가 한 step에 크게 튀는 것 자체는 막을 수 없다.

두 문제의 공통 원인: **e_neg가 갑자기 extreme한 값으로 가는 것**. CD loss = E(demo) - E(negative)이고, e_neg가 크게 음수가 되면 CD loss 부호가 뒤집히고 gradient가 폭발한다.

### 이번 변경: energy_clamp=20.0

`e_pos`와 `e_neg`를 backward 직전에 `clamp(-20, +20)`으로 제한한다.

정상 학습 구간에서 e+≈-5, e-≈+5이므로 20.0은 평상시에 거의 걸리지 않는다. e_neg가 갑자기 -10이나 -20 방향으로 튀는 순간에만 개입해서 loss가 뒤집히는 것을 막는다.

**6.0으로 먼저 시도했다가 실패했다**: EBM 초기화 시 출력값이 -30 ~ -50 수준이라, 6.0으로 clamped하면 e+와 e- 둘 다 -6.0에 고착되어 CD loss=0, ∇rw=0이 됐다. gradient가 완전히 죽어서 즉시 중단했다.

### 합격 기준

| 항목 | 기준 | 이유 |
|------|------|------|
| best val ρ | > 0.2791 (v1 best) | clamp이 도움이 됐다는 직접 증거 |
| ep20 같은 single-epoch collapse | 없음 | 불안정 요소 제거 확인 |
| ∇rw spike (>100) | 횟수 감소 | clamp이 CD loss 안정화에 기여 확인 |
| sep_std_ema (FM 활성 후) | 계속 업데이트됨 | v1 구조 버그(freeze) 수정 확인 |

---

## 3. 학습 손실 곡선

*(학습 완료 후 작성)*

---

## 4. 검증 지표

*(학습 완료 후 작성)*

### v1 대비 비교 (예정)

| epoch | v1 ρ | v2 ρ | 개선 |
|-------|------|------|------|
| ... | ... | ... | ... |

---

## 5. 결과 해석 및 인사이트

*(학습 완료 후 작성)*

### 미리 세워두는 해석 기준

결과가 나오면 아래 세 가지 시나리오 중 하나가 될 것이다.

**시나리오 A — clamp 효과 있음**: ep20 같은 collapse가 사라지고, ∇rw spike 횟수가 줄고, best val ρ > 0.2791. 이 경우 "energy_clamp이 FM hybrid 학습의 안정화에 기여한다"는 주장이 가능해진다. 다음 단계는 seed 2, 3 재현으로 넘어간다.

**시나리오 B — clamp 효과 없음 (안정화는 됐지만 성능 동일)**: ∇rw spike는 줄었지만 best val ρ가 v1과 비슷한 수준. 이 경우 "clamp이 stability는 높이지만 ranking 성능 자체는 다른 요인에 달려있다"는 뜻이다. 여전히 유용한 결과 — seed 재현 시 분산이 줄어드는지 확인.

**시나리오 C — clamp이 오히려 방해**: best val ρ가 v1보다 낮다. clamp이 gradient flow를 제한해서 EBM의 학습 속도를 떨어뜨렸을 가능성. 이 경우 energy_clamp 방향은 버리고 gradient 관련 다른 접근(e.g., adaptive grad clip, loss normalization)을 검토한다.

---

## 6. 다음 스텝

*(학습 완료 후 업데이트)*

---

## 7. 저장 파일 목록

```
outputs/ebm_fm_gate_v2_20260422/
├── train.log           # 학습 로그 (자동 생성)
├── RESULT.md           # 이 파일
├── ckpt_best_val.pt    # best val checkpoint — git 제외
└── ckpt_epoch0030.pt   # last checkpoint — git 제외
```
