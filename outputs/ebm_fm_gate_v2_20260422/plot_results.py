"""
ebm_fm_gate_v2 실험 결과 시각화
train.log 파싱 → 학습 과정 그래프 생성
v1 대비 변경: energy_clamp=20.0 + sep monitoring fix
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

LOG_PATH = Path(__file__).parent / "train.log"
OUT_PATH = Path(__file__).parent / "training_curves.png"

# ── 파싱 ──────────────────────────────────────────────────────────────────────

step_re = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\] \[ep(\d+)\|s(\d+)\]"
    r"\s+rw=([+-]?\d+\.\d+)\s+cd=([+-]?\d+\.\d+)\s+reg=([+-]?\d+\.\d+)"
    r".*?e\+=([ +\-\d.]+)\(±([\d.]+)\)\s+e-=([ +\-\d.]+)\(±([\d.]+)\)"
    r".*?sep=([+-]?[\d.]+)"
    r".*?fm_e=([+-]?[\d.]+|nan)\s+fm=(ON|off)"
    r".*?∇rw=([\d.]+)\s+∇pol=([\d.]+)"
)
val_re = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\] ══+ VAL ep(\d+) ══+"
    r"\s+ρ=([+-]?\d+\.\d+).*?p=([\d.]+).*?\[(PASS|FAIL)"
    r".*?AUROC\(E\)=([\d.]+)\s+ECE=([\d.]+)"
)

steps, vals = [], []

with open(LOG_PATH) as f:
    lines = f.readlines()

# 두 번째 run 시작점 찾기 (첫 번째 run은 초기화 상태로 조기 종료)
epoch1_starts = [i for i, l in enumerate(lines) if "Epoch 01/30" in l]
start_line = epoch1_starts[1] if len(epoch1_starts) >= 2 else 0

for line in lines[start_line:]:
    m = step_re.search(line)
    if m:
        fm_e_raw = m.group(12)
        steps.append({
            "ep": int(m.group(2)), "step": int(m.group(3)),
            "rw":  float(m.group(4)),
            "cd":  float(m.group(5)),
            "reg": float(m.group(6)),
            "e_pos":     float(m.group(7)),
            "e_pos_std": float(m.group(8)),
            "e_neg":     float(m.group(9)),
            "e_neg_std": float(m.group(10)),
            "sep":       float(m.group(11)),
            "fm_e":  float("nan") if fm_e_raw == "nan" else float(fm_e_raw),
            "fm_on": m.group(13) == "ON",
            "grad_rw":  float(m.group(14)),
            "grad_pol": float(m.group(15)),
        })
        continue
    m = val_re.search(line)
    if m:
        vals.append({
            "ep":    int(m.group(2)),
            "rho":   float(m.group(3)),
            "p":     float(m.group(4)),
            "pass":  m.group(5) == "PASS",
            "auroc": float(m.group(6)),
            "ece":   float(m.group(7)),
        })

steps = sorted(steps, key=lambda x: x["step"])
vals  = sorted(vals,  key=lambda x: x["ep"])

gate_step = next((d["step"] for d in steps if d["fm_on"]), None)
gate_ep   = next((d["ep"]   for d in steps if d["fm_on"]), None)

s_arr    = np.array([d["step"]      for d in steps])
rw_arr   = np.array([d["rw"]        for d in steps])
cd_arr   = np.array([d["cd"]        for d in steps])
reg_arr  = np.array([d["reg"]       for d in steps])
epos_arr = np.array([d["e_pos"]     for d in steps])
epos_s   = np.array([d["e_pos_std"] for d in steps])
eneg_arr = np.array([d["e_neg"]     for d in steps])
eneg_s   = np.array([d["e_neg_std"] for d in steps])
sep_arr  = np.array([d["sep"]       for d in steps])
fme_arr  = np.array([d["fm_e"]      for d in steps])
grw_arr  = np.array([d["grad_rw"]   for d in steps])
gpol_arr = np.array([d["grad_pol"]  for d in steps])
fmon_arr = np.array([d["fm_on"]     for d in steps])

v_ep   = np.array([d["ep"]    for d in vals])
v_rho  = np.array([d["rho"]   for d in vals])
v_auc  = np.array([d["auroc"] for d in vals])
v_ece  = np.array([d["ece"]   for d in vals])
v_pass = np.array([d["pass"]  for d in vals])
best_i = int(np.argmax(v_rho))

# ── 플롯 ──────────────────────────────────────────────────────────────────────

GATE_COLOR = "#e67e22"
PASS_COLOR = "#27ae60"
FAIL_COLOR = "#e74c3c"
ALPHA_BAND = 0.15

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle(
    f"ebm_fm_gate_v2_20260422  (seed=1, 30 epochs, energy_clamp=20.0 + sep monitoring fix)\n"
    f"FM gate opened @ ep{gate_ep}/s{gate_step}  │  best val ρ=+0.2848 @ ep29",
    fontsize=13, fontweight="bold", y=0.98
)

def add_gate_v(ax):
    if gate_step:
        ax.axvline(gate_step, color=GATE_COLOR, lw=1.2, ls="--", alpha=0.8,
                   label=f"FM gate open (s{gate_step})")

def add_gate_ep(ax):
    if gate_ep:
        ax.axvline(gate_ep, color=GATE_COLOR, lw=1.2, ls="--", alpha=0.8,
                   label=f"FM gate open (ep{gate_ep})")

def smooth(arr, w=5):
    return np.convolve(arr, np.ones(w) / w, mode="same")

# ── (0,0) Training Loss ───────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(s_arr, rw_arr, color="#3498db", alpha=0.25, lw=0.7)
ax.plot(s_arr, smooth(rw_arr), color="#3498db", lw=1.8, label="reward loss")
ax.plot(s_arr, cd_arr, color="#9b59b6", alpha=0.25, lw=0.7)
ax.plot(s_arr, smooth(cd_arr), color="#9b59b6", lw=1.8, label="CD loss")
ax.plot(s_arr, reg_arr, color="#95a5a6", alpha=0.25, lw=0.7)
ax.plot(s_arr, smooth(reg_arr), color="#95a5a6", lw=1.8, label="reg loss")
add_gate_v(ax)
ax.set_xlabel("step"); ax.set_ylabel("loss")
ax.set_title("Training Loss")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── (0,1) Val Spearman ρ ──────────────────────────────────────────────────────
ax = axes[0, 1]
ax.axhline(0.20,  color="gray",    ls=":", lw=1,   alpha=0.6, label="ρ=0.20 threshold")
ax.axhline(0.239, color="#2980b9", ls=":", lw=1.2, alpha=0.7, label="C avg test (0.239)")
ax.axhline(0.208, color="#7f8c8d", ls=":", lw=1,   alpha=0.6, label="B test (0.208)")
ax.plot(v_ep, v_rho, color="#2c3e50", lw=1.5, alpha=0.6)
ax.scatter(v_ep[v_pass],  v_rho[v_pass],  color=PASS_COLOR, s=55, zorder=5, label="PASS")
ax.scatter(v_ep[~v_pass], v_rho[~v_pass], color=FAIL_COLOR, s=55, zorder=5, marker="x", label="FAIL")
ax.scatter(v_ep[best_i], v_rho[best_i], color="gold", s=140, zorder=6,
           edgecolors="black", lw=1, label=f"best ep{v_ep[best_i]} ρ={v_rho[best_i]:.4f}")
add_gate_ep(ax)
ax.set_xlabel("epoch"); ax.set_ylabel("Spearman ρ")
ax.set_title("Val Spearman ρ")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── (1,0) Energy Separation ───────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(s_arr, epos_arr, color=PASS_COLOR, lw=1.8, label="E+ (demo)")
ax.fill_between(s_arr, epos_arr - epos_s, epos_arr + epos_s,
                color=PASS_COLOR, alpha=ALPHA_BAND)
ax.plot(s_arr, eneg_arr, color=FAIL_COLOR, lw=1.8, label="E− (SGLD neg)")
ax.fill_between(s_arr, eneg_arr - eneg_s, eneg_arr + eneg_s,
                color=FAIL_COLOR, alpha=ALPHA_BAND)
# FM neg energy (when FM is ON)
fme_valid = np.where(fmon_arr & ~np.isnan(fme_arr), fme_arr, np.nan)
ax.plot(s_arr, fme_valid, color="#e67e22", lw=1.2, alpha=0.7, ls="--", label="fm_e (FM neg)")
add_gate_v(ax)
ax.set_xlabel("step"); ax.set_ylabel("energy")
ax.set_title("Energy Separation  (E+ vs E−, ±1σ)  +  FM sample energy")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── (1,1) Val AUROC(E) & ECE ─────────────────────────────────────────────────
ax = axes[1, 1]
ax2 = ax.twinx()
ax.plot(v_ep, v_auc, color="#2980b9", lw=1.8, marker="o", ms=4, label="AUROC(E)")
ax.axhline(0.688, color="#2980b9", ls=":", lw=1.2, alpha=0.7, label="C avg AUROC (0.688)")
ax2.plot(v_ep, v_ece, color=GATE_COLOR, lw=1.8, marker="s", ms=4, alpha=0.8, label="ECE")
ax.set_xlabel("epoch")
ax.set_ylabel("AUROC(E)", color="#2980b9")
ax2.set_ylabel("ECE", color=GATE_COLOR)
ax.set_title("Val AUROC(E) & ECE")
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
ax.grid(alpha=0.3)

# ── (2,0) Reward Grad Norm ────────────────────────────────────────────────────
ax = axes[2, 0]
ax.semilogy(s_arr, grw_arr, color="#e74c3c", alpha=0.3, lw=0.6)
ax.semilogy(s_arr, smooth(grw_arr, w=9), color="#e74c3c", lw=1.8, label="∇rw (smoothed)")
# v1 참고선 — 최대 spike 수준
ax.axhline(624, color="#c0392b", ls=":", lw=1, alpha=0.5, label="v1 max spike (624)")
ax.axhline(20,  color="gray",    ls=":", lw=1, alpha=0.5, label="threshold=20")
add_gate_v(ax)
ax.set_xlabel("step"); ax.set_ylabel("grad norm (log)")
ax.set_title("Reward Grad Norm  ∇rw  (vs v1 max spike)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# ── (2,1) sep_std_ema trajectory ─────────────────────────────────────────────
ax = axes[2, 1]
# sep=-1.0은 gate 이전 미계산 구간 — 제외
sep_valid_mask = sep_arr > 0
ax.plot(s_arr[sep_valid_mask], sep_arr[sep_valid_mask],
        color="#8e44ad", alpha=0.4, lw=0.7)
ax.plot(s_arr[sep_valid_mask], smooth(sep_arr[sep_valid_mask], w=7),
        color="#8e44ad", lw=1.8, label="sep_std_ema (smoothed)")
ax.axhline(40, color=GATE_COLOR, ls="--", lw=1.2, alpha=0.8,
           label="fm_gate threshold (40)")
ax.fill_between(s_arr, 0, 1, where=fmon_arr,
                color=GATE_COLOR, alpha=0.08, transform=ax.get_xaxis_transform(),
                label="FM ON region")
add_gate_v(ax)
ax.set_xlabel("step"); ax.set_ylabel("sep_std_ema")
ax.set_title("sep_std_ema Trajectory  (monitoring fix 검증)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── (3,0) FM Gate Timeline & Energy Separation ────────────────────────────────
ax = axes[3, 0]
ax.fill_between(s_arr, 0, 1, where=~fmon_arr,
                color="#bdc3c7", alpha=0.5, transform=ax.get_xaxis_transform(),
                label="Phase 1: SGLD-only")
ax.fill_between(s_arr, 0, 1, where=fmon_arr,
                color=GATE_COLOR, alpha=0.4, transform=ax.get_xaxis_transform(),
                label="Phase 2: FM hybrid ON")
ax.plot(s_arr, np.abs(epos_arr - eneg_arr), color="#2c3e50", lw=1.5,
        label="|E+ − E−| (separation)")
add_gate_v(ax)
ax.set_xlabel("step"); ax.set_ylabel("|E+ − E−|")
ax.set_title("FM Gate Timeline & Energy Separation")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── (3,1) Val ρ vs Previous Experiments ──────────────────────────────────────
ax = axes[3, 1]
ax.axhspan(-0.15, 0.0,  color=FAIL_COLOR, alpha=0.05)
ax.axhspan(0.0,   0.20, color="khaki",    alpha=0.10)
ax.axhspan(0.20,  0.35, color=PASS_COLOR, alpha=0.07)
ax.plot(v_ep, v_rho, color="#2c3e50", lw=2)
refs = [
    ("C r1 best (0.236)", 0.2364, "#aab7b8"),
    ("C r2 best (0.290)", 0.2902, "#aab7b8"),
    ("C r3 best (0.295)", 0.2946, "#aab7b8"),
    ("B test (0.208)",    0.2083, "#2980b9"),
    ("v1 best (0.279)",   0.2791, "#e67e22"),
]
for label, val, col in refs:
    ax.axhline(val, ls="--", lw=1, color=col, alpha=0.7, label=label)
ax.scatter(v_ep[v_pass],  v_rho[v_pass],  color=PASS_COLOR, s=50, zorder=5)
ax.scatter(v_ep[~v_pass], v_rho[~v_pass], color=FAIL_COLOR, s=50, zorder=5, marker="x")
ax.scatter(v_ep[best_i], v_rho[best_i], color="gold", s=160, zorder=6,
           edgecolors="black", lw=1.2)
ax.text(v_ep[best_i] + 0.3, v_rho[best_i] + 0.005,
        f"ep{v_ep[best_i]}\nρ={v_rho[best_i]:.4f}", fontsize=8, fontweight="bold")
add_gate_ep(ax)
ax.set_xlabel("epoch"); ax.set_ylabel("Spearman ρ")
ax.set_title("Val ρ vs. Previous Experiments")
ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"saved → {OUT_PATH}")
