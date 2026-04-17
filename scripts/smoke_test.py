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


# ── Test 3: rollout() clamp saturation rate ────────────────────────────────────

def test_clamp_saturation(vf, irl):
    print("\n[Test 3] rollout() clamp saturation rate")
    from src.models.flow_matching import rollout

    x0 = torch.randn(4, *PATCH)
    with torch.no_grad():
        x1 = rollout(vf, x0, n_steps=32)

    clamped = ((x1 <= 0.0) | (x1 >= 1.0)).float().mean().item()
    # 기준: 30% 이상이면 경계 포화 심각
    ok = clamped < 0.30

    print_result(
        f"clamp saturation rate < 30%  (got {clamped:.1%})",
        ok,
    )

    if not ok:
        print(f"    {WARN} 경계 포화율 높음 → reward_term_grad_norm이 낮을 가능성 큼")
        print(f"    → rollout() 내 .clamp(0,1) 제거 또는 tanh 스케일링 검토")

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


# ── 종합 ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Smoke Test: grad path + clamp + negative quality")
    print("=" * 60)

    ebm, vf, irl = make_models()

    r1 = test_reward_grad(ebm, vf, irl)
    r2 = test_policy_grad(ebm, vf, irl)
    r3_ok, clamp_rate = test_clamp_saturation(vf, irl)
    r4_ok, neg_ratio  = test_negative_quality(ebm, irl)

    print("\n" + "=" * 60)
    print("종합 결과")
    print("-" * 60)
    print_result("reward grad path",         r1)
    print_result("policy grad path",         r2)
    print_result(f"clamp rate ({clamp_rate:.1%})",  r3_ok)
    print_result(f"negative quality (ratio={neg_ratio:.1f})", r4_ok)

    all_pass = r1 and r2 and r3_ok and r4_ok
    print()
    if all_pass:
        print("  전체 PASS — saturation mini-run 진행 가능")
    else:
        print("  FAIL 항목 수정 후 재실행. Full retrain 금지.")
        if not r3_ok:
            print("  → 우선순위: rollout() clamp 수정")
        if not r4_ok:
            print("  → 우선순위: FM trivial negative 대응 (SGLD hybrid)")
        if not r2:
            print("  → 우선순위: reward_term_grad_norm 원인 추적")

    print("=" * 60)


if __name__ == "__main__":
    main()
