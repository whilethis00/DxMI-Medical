# April Week 3 (2026-04-13 ~ 2026-04-19)

## 현재 상태 파악 (2026-04-17 기준)

---

## 이전 주(W2) 마무리 요약

### Ablation 실험 결과 (validation, N=177)

| 실험 | 설명 | Spearman ρ | p-value | AUROC | ECE | 판정 |
|------|------|------------|---------|-------|-----|------|
| A | No IRL | -0.0814 | 0.2814 | 0.4758 | 0.7956 | FAIL |
| B | Supervised reward | 0.2455 | 0.0010 | 0.3261 | 0.1989 | PASS |
| C | MaxEnt IRL | 0.0778 | 0.3034 | 0.5812 | 0.7955 | FAIL |

- 목표(`A < B < C`) 미달성, B만 clinical 기준(p < 0.05) 통과
- A/C 모두 `e_pos≈-5`, `e_neg≈5`로 빠르게 포화되는 패턴

---

## Apr 14 코드 수정 현황 (미커밋)

`git diff HEAD` 기준 8개 파일 수정, 총 +716 / -240 라인. **아직 커밋 안 됨.**

### `src/models/irl.py` — Priority 1/2 수정 진행 중

- `_policy_sample()` 추가: FM rollout (detach/grad 모드 분리)
- `update_reward()`: SGLD negative → FM policy sample 기반으로 교체
  - demo vs policy-generated sample 직접 비교 구조
- `_freeze_ebm_params()` context manager 추가
  - policy loss 계산 시 EBM 파라미터 freeze, gradient는 FM쪽으로만 흐르도록
- `_grad_norm()` 추가: reward/policy gradient norm 로깅용
- DDP 래퍼 지원 (`_raw_ebm`, `_raw_vf` 분리)
- `IRLConfig`에 `policy_sample_steps`, `policy_grad_steps`, `reward_weight` 추가
- 추가 로그 항목: `cd_loss`, `reg_loss`, `e_pos_std`, `e_neg_std`, `reward_grad_norm`, `fm_sample_energy`

### `src/models/flow_matching.py`

- `rollout()` 함수 추가 (Euler integration, `_policy_sample`에서 사용)

### `src/models/ebm.py`

- Priority 3 (EBM loss 재설계) 일부 수정

### `scripts/train.py`

- 로깅 강화 (Priority 6)
- DDP 지원 (torchrun / DistributedSampler) — W2 말에 이미 커밋됨

### `configs/ebm_baseline.yaml`, `configs/irl_maxent.yaml`

- 하이퍼파라미터 조정 (`l2_reg`, `sgld_step_size` 등)

### `src/evaluation/metrics.py`

- AUROC score sign 검토 (Priority 5)

---

## 남은 작업

### 즉시 해야 할 것

- [ ] Apr 14 수정분 코드 리뷰 완료 및 커밋
- [ ] smoke test (CPU, 2-step) — gradient path 검증
  - `reward_loss.backward()` 시 `VelocityField` 파라미터에 gradient 흐르는지 확인
  - A/C reward dynamics가 이전과 다른 패턴을 보이는지 확인

### 재실험

- [ ] GPU 잡 확보 후 Ablation A/C 재훈련 (수정된 코드 기준)
- [ ] `l2_reg`, `sgld_steps`, `sgld_step_size` 재탐색 (Priority 4)
- [ ] 재실험 결과로 `A < B < C` 달성 여부 확인

### 결과 정리 (재실험 후)

- [ ] test set 평가 및 val/test 통합 결과표 작성
- [ ] 논문용 figure/table 정리
- [ ] 실패 원인 및 수정 계획 문서화

---

## 목표 지표 (재확인)

| 실험 | 합격 조건 |
|------|----------|
| A < B | baseline 비교 |
| B < C | 제안 방법 우위 |
| Clinical | Spearman ρ(energy, disagreement), p < 0.05 |

---

## 참고

- 타겟: ICLR 2027 (제출 마감 2026년 9월 말)
- 베이스 논문: DxMI (NeurIPS 2024)
- 코드 수정 우선순위: `Code_Fix_Plan.md`, `Code_Fix_Priority.md` 참조
