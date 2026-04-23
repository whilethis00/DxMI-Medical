"""
Gradient path + clamp saturation + FM negative quality smoke test

검증 항목:
  1. update_reward(): EBM에 grad 있음, VF에 grad 없음
  2. update_policy(): VF에 grad 있음 (fm_loss 경로 + reward 경로 각각)
                      reward_term_grad_norm > 0 확인
  3. rollout() clamp saturation: boundary에 걸린 비율 계측
  4. FM policy sample negative quality: demo vs policy sample energy 분포 비교

CPU 2-step으로 충분. GPU 없어도 돌아간다.

Usage:
    cd /home/introai26/.agile/user/hsjung/DxMI_Medical
    /home/introai26/miniconda3/envs/dxmi_medical/bin/python scripts/smoke_test.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.ebm import EBM
from src.models.flow_matching import VelocityField
from src.models.irl import MaxEntIRL, IRLConfig

DEVICE = torch.device("cpu")
B = 2         # 배치 크기 — smoke test용 최소값
PATCH = (1, 48, 48, 48)

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def make_models():
    # 작은 채널 수로 빠르게
    ebm = EBM(base_ch=8).to(DEVICE)
    vf  = VelocityField(base_ch=8).to(DEVICE)
    cfg = IRLConfig(
        reward_steps_per_iter=1,
        fm_steps_per_iter=1,
        sgld_steps=2,
        policy_sample_steps=2,   # 빠른 rollout
        policy_grad_steps=2,
    )
    irl = MaxEntIRL(ebm, vf, cfg, DEVICE)
    return ebm, vf, irl


def zero_grads(models):
    for m in models:
        for p in m.parameters():
            p.grad = None


def has_grad(module: nn.Module) -> bool:
    return any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in module.parameters()
    )


def print_result(label: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ── Test 1: update_reward grad destination ─────────────────────────────────────

def test_reward_grad(ebm, vf, irl):
    print("\n[Test 1] update_reward() grad destination")
    x_demo = torch.randn(B, *PATCH)

    zero_grads([ebm, vf])
    irl.update_reward(x_demo)

    ebm_ok = has_grad(ebm)
    vf_ok  = not has_grad(vf)

    print_result("EBM params have grad after update_reward", ebm_ok)
    print_result("VF params have NO grad after update_reward", vf_ok)

    if not vf_ok:
        print(f"    {WARN} VF got unexpected grad — check reward_opt / optimizer scope")

    return ebm_ok and vf_ok


# ── Test 2: update_policy grad destination + reward path ──────────────────────

def test_policy_grad(ebm, vf, irl):
    print("\n[Test 2] update_policy() grad destination + reward path")
    x_demo = torch.randn(B, *PATCH)

    zero_grads([ebm, vf])
    metrics = irl.update_policy(x_demo)

    vf_ok  = has_grad(vf)
    ebm_ok = not has_grad(ebm)

    reward_norm = metrics.get("reward_term_grad_norm", 0.0)
    policy_norm = metrics.get("policy_term_grad_norm", 0.0)

    print_result("VF params have grad after update_policy", vf_ok)
    print_result("EBM params have NO grad after update_policy", ebm_ok)
    print_result(
        f"reward_term_grad_norm > 0  (got {reward_norm:.2e})",
        reward_norm > 1e-12,
    )
    print_result(
        f"policy_term_grad_norm > 0  (got {policy_norm:.2e})",
        policy_norm > 1e-12,
    )

    if reward_norm <= 1e-12:
        print(f"    {WARN} reward path killed — likely clamp saturation or graph断절")

    if not ebm_ok:
        print(f"    {WARN} EBM grad leak — check _freeze_ebm_params()")

    return vf_ok and (reward_norm > 1e-12)


# ── Test 3: rollout() gradient flow through all steps ─────────────────────────

def test_clamp_saturation(vf, irl):
    print("\n[Test 3] rollout() gradient flow (intermediate clamp 제거 검증)")
    from src.models.flow_matching import rollout

    # 32-step rollout에서 VF 파라미터까지 gradient가 도달하는지 직접 확인
    x0 = torch.randn(4, *PATCH)
    for p in vf.parameters():
        p.grad = None

    x1 = rollout(vf, x0, n_steps=32)
    x1.sum().backward()

    grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in vf.parameters()
        if p.grad is not None
    ) ** 0.5
    ok = grad_norm > 1e-6

    print_result(
        f"grad flows through 32-step rollout to VF params  (norm={grad_norm:.2e})",
        ok,
    )

    # 출력 saturation 비율은 정보 표시만 (random model에서 높은 것은 정상)
    with torch.no_grad():
        x1_eval = rollout(vf, x0, n_steps=32)
    clamped = ((x1_eval <= 0.0) | (x1_eval >= 1.0)).float().mean().item()
    print(f"    output saturation rate: {clamped:.1%}  (random model 기준, 학습 후 감소 예상)")

    if not ok:
        print(f"    {WARN} gradient가 rollout 통해 VF에 미달 — intermediate clamp 확인")

    return ok, clamped


# ── Test 4: FM policy sample negative quality ──────────────────────────────────

def test_negative_quality(ebm, irl):
    print("\n[Test 4] FM policy sample negative quality")
    from src.models.flow_matching import rollout

    x_demo = torch.rand(4, *PATCH)     # 실제 데이터를 흉내낸 uniform [0,1] 패치

    # policy sample (early training ≈ noise)
    x0 = torch.randn(4, *PATCH)
    with torch.no_grad():
        x_policy = rollout(irl._raw_vf, x0, n_steps=32)

    with torch.no_grad():
        e_demo   = ebm(x_demo)
        e_policy = ebm(x_policy)

    sep = (e_demo.mean() - e_policy.mean()).abs().item()
    e_demo_std   = e_demo.std().item()
    e_policy_std = e_policy.std().item()

    print(f"    demo   energy: mean={e_demo.mean():.3f}  std={e_demo_std:.3f}")
    print(f"    policy energy: mean={e_policy.mean():.3f}  std={e_policy_std:.3f}")
    print(f"    separation (|mean_demo - mean_policy|): {sep:.3f}")

    # 기준: separation이 std보다 훨씬 크면 trivially easy separation
    avg_std = (e_demo_std + e_policy_std) / 2 + 1e-8
    ratio   = sep / avg_std

    trivial = ratio > 3.0   # separation이 std의 3배 이상이면 trivial negative 의심
    print_result(
        f"separation/std ratio < 3  (got {ratio:.2f}) — trivial negative 여부",
        not trivial,
    )

    if trivial:
        print(f"    {WARN} FM policy sample이 너무 쉬운 negative일 가능성")
        print(f"    → SGLD hybrid 또는 warm-up 기간 동안 SGLD 사용 검토")

    return not trivial, ratio


# ── Test 5: FM quality filter (v3) ────────────────────────────────────────────

def test_fm_quality_filter(ebm, vf):
    print("\n[Test 5] FM quality filter — fallback path + metric correctness")

    def _make_irl_with_filter(threshold):
        cfg = IRLConfig(
            reward_steps_per_iter=1,
            fm_steps_per_iter=1,
            sgld_steps=2,
            policy_sample_steps=2,
            policy_grad_steps=2,
            fm_quality_filter=True,
            fm_quality_threshold=threshold,
        )
        irl = MaxEntIRL(ebm, vf, cfg, DEVICE)
        irl._fm_enabled = True  # gate를 이미 열린 상태로 강제 설정
        return irl

    x_demo = torch.rand(B, *PATCH)
    all_ok = True

    # 5-a: threshold=+999 → 어떤 FM 에너지도 999 미만 → 항상 fallback
    irl_high = _make_irl_with_filter(threshold=+999.0)
    x_neg_fb, src_fb = irl_high._sample_negatives(x_demo)
    fb_ok = (src_fb == "sgld_fallback")
    print_result(
        f"threshold=+999 → neg_source='sgld_fallback'  (got '{src_fb}')",
        fb_ok,
    )
    shape_fb = (x_neg_fb.shape == (B, 1, 48, 48, 48))
    print_result(
        f"fallback x_neg shape == (B,1,48,48,48)  (got {tuple(x_neg_fb.shape)})",
        shape_fb,
    )
    all_ok = all_ok and fb_ok and shape_fb

    # 5-b: threshold=-999 → 어떤 FM 에너지도 -999 이상 → fallback 안 함
    irl_low = _make_irl_with_filter(threshold=-999.0)
    x_neg_hy, src_hy = irl_low._sample_negatives(x_demo)
    hy_ok = (src_hy == "hybrid")
    print_result(
        f"threshold=-999 → neg_source='hybrid'  (got '{src_hy}')",
        hy_ok,
    )
    all_ok = all_ok and hy_ok

    # 5-c: update_reward metrics에 fm_fallback_rate 키 존재 + 값 확인
    metrics_fb = irl_high.update_reward(x_demo)
    key_ok  = "fm_fallback_rate" in metrics_fb
    rate_ok = metrics_fb.get("fm_fallback_rate", -1.0) == 1.0
    print_result(
        "'fm_fallback_rate' key in update_reward metrics",
        key_ok,
    )
    print_result(
        f"fm_fallback_rate == 1.0 when always fallback  (got {metrics_fb.get('fm_fallback_rate', 'N/A')})",
        rate_ok,
    )
    all_ok = all_ok and key_ok and rate_ok

    metrics_hy = irl_low.update_reward(x_demo)
    rate_hy_ok = metrics_hy.get("fm_fallback_rate", -1.0) == 0.0
    print_result(
        f"fm_fallback_rate == 0.0 when never fallback  (got {metrics_hy.get('fm_fallback_rate', 'N/A')})",
        rate_hy_ok,
    )
    all_ok = all_ok and rate_hy_ok

    # 5-d: fallback 후에도 EBM grad path 정상 (update_reward 후 EBM에 grad 있어야 함)
    for p in ebm.parameters():
        p.grad = None
    irl_high.update_reward(x_demo)
    ebm_grad_ok = has_grad(ebm)
    print_result(
        "EBM gets grad after update_reward with fallback",
        ebm_grad_ok,
    )
    all_ok = all_ok and ebm_grad_ok

    return all_ok


# ── 종합 ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Smoke Test: grad path + clamp + negative quality + FM filter")
    print("=" * 60)

    ebm, vf, irl = make_models()

    r1 = test_reward_grad(ebm, vf, irl)
    r2 = test_policy_grad(ebm, vf, irl)
    r3_ok, clamp_rate = test_clamp_saturation(vf, irl)
    r4_ok, neg_ratio  = test_negative_quality(ebm, irl)
    r5 = test_fm_quality_filter(ebm, vf)

    print("\n" + "=" * 60)
    print("종합 결과")
    print("-" * 60)
    print_result("reward grad path",                       r1)
    print_result("policy grad path",                       r2)
    print_result(f"clamp rate ({clamp_rate:.1%})  [기존 문제]", r3_ok)
    print_result(f"negative quality (ratio={neg_ratio:.1f})", r4_ok)
    print_result("FM quality filter (v3)",                 r5)

    all_pass = r1 and r2 and r3_ok and r4_ok and r5
    v3_pass  = r1 and r2 and r5        # v3 핵심 항목만
    print()
    if all_pass:
        print("  전체 PASS — saturation mini-run 진행 가능")
    elif v3_pass:
        print("  v3 핵심 항목 PASS (clamp FAIL은 기존 문제, v3 무관)")
        print("  → push 진행 가능")
    else:
        print("  FAIL 항목 수정 후 재실행. Full retrain 금지.")
        if not r3_ok:
            print("  → rollout() clamp 수정 (기존 문제)")
        if not r4_ok:
            print("  → FM trivial negative 대응 (SGLD hybrid)")
        if not r2:
            print("  → reward_term_grad_norm 원인 추적")
        if not r5:
            print("  → FM quality filter 로직 수정 (v3 핵심)")

    print("=" * 60)


if __name__ == "__main__":
    main()
