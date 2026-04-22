"""
학습 결과 자동 시각화 스크립트.
train.py 완료 시 자동 호출되거나 단독 실행 가능.

Usage:
    python scripts/plot_training.py --output_dir outputs/ebm_fm_gate_v2_20260422
"""

import re
import sys
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 로그 파싱 ──────────────────────────────────────────────────────────────────

STEP_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\] \[ep(\d+)\|s(\d+)\]"
    r"\s+rw=([+-]?\d+\.\d+)\s+cd=([+-]?\d+\.\d+)\s+reg=([+-]?\d+\.\d+)"
    r".*?e\+=([ +\-\d.]+)\(±([\d.]+)\)\s+e-=([ +\-\d.]+)\(±([\d.]+)\)"
    r".*?fm_e=([+-]?[\d.]+|nan)\s+fm=(ON|off)"
    r".*?∇rw=([\d.]+)\s+∇pol=([\d.]+)"
)
VAL_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\] ══+ VAL ep(\d+) ══+"
    r"\s+ρ=([+-]?\d+\.\d+).*?p=([\d.]+).*?\[(PASS|FAIL)"
    r".*?AUROC\(E\)=([\d.]+)\s+ECE=([\d.]+)"
)


def parse_log(log_path: Path):
    steps, vals = [], []
    lines = log_path.read_text().splitlines()

    # run이 여러 번 있으면 마지막 run만 사용 (재시작 실험 대응)
    epoch1_starts = [i for i, l in enumerate(lines) if "Epoch 01/30" in l
                     or "Epoch 01/" in l]
    start = epoch1_starts[-1] if epoch1_starts else 0

    for line in lines[start:]:
        m = STEP_RE.search(line)
        if m:
            steps.append({
                "ep":        int(m.group(2)),
                "step":      int(m.group(3)),
                "rw":        float(m.group(4)),
                "cd":        float(m.group(5)),
                "reg":       float(m.group(6)),
                "e_pos":     float(m.group(7)),
                "e_pos_std": float(m.group(8)),
                "e_neg":     float(m.group(9)),
                "e_neg_std": float(m.group(10)),
                "fm_e":      float("nan") if m.group(11) == "nan" else float(m.group(11)),
                "fm_on":     m.group(12) == "ON",
                "grad_rw":   float(m.group(13)),
                "grad_pol":  float(m.group(14)),
            })
            continue

        m = VAL_RE.search(line)
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
    return steps, vals


# ── 플롯 생성 ──────────────────────────────────────────────────────────────────

GATE_COLOR = "#e67e22"
PASS_COLOR = "#27ae60"
FAIL_COLOR = "#e74c3c"
ALPHA_BAND = 0.15


def smooth(arr, w=5):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


def generate_plot(output_dir: Path, save_path: Path | None = None):
    log_path = output_dir / "train.log"
    if not log_path.exists():
        print(f"[plot] train.log not found: {log_path}")
        return

    steps, vals = parse_log(log_path)
    if not steps:
        print("[plot] no step data parsed, skipping")
        return

    if save_path is None:
        save_path = output_dir / "training_curves.png"

    exp_name  = output_dir.name
    gate_step = next((d["step"] for d in steps if d["fm_on"]), None)
    gate_ep   = next((d["ep"]   for d in steps if d["fm_on"]), None)
    best_rho  = max((d["rho"] for d in vals), default=float("nan"))
    best_ep   = next((d["ep"] for d in vals if d["rho"] == best_rho), None)
    n_epochs  = max((d["ep"] for d in steps), default="?")

    s_arr    = np.array([d["step"]      for d in steps])
    rw_arr   = np.array([d["rw"]        for d in steps])
    cd_arr   = np.array([d["cd"]        for d in steps])
    reg_arr  = np.array([d["reg"]       for d in steps])
    epos_arr = np.array([d["e_pos"]     for d in steps])
    epos_s   = np.array([d["e_pos_std"] for d in steps])
    eneg_arr = np.array([d["e_neg"]     for d in steps])
    eneg_s   = np.array([d["e_neg_std"] for d in steps])
    grw_arr  = np.array([d["grad_rw"]   for d in steps])
    gpol_arr = np.array([d["grad_pol"]  for d in steps])
    fm_mask  = np.array([d["fm_on"]     for d in steps])

    v_ep   = np.array([d["ep"]    for d in vals]) if vals else np.array([])
    v_rho  = np.array([d["rho"]   for d in vals]) if vals else np.array([])
    v_auc  = np.array([d["auroc"] for d in vals]) if vals else np.array([])
    v_ece  = np.array([d["ece"]   for d in vals]) if vals else np.array([])
    v_pass = np.array([d["pass"]  for d in vals]) if vals else np.array([], dtype=bool)

    # title 구성
    gate_info = f"FM gate @ ep{gate_ep}/s{gate_step}" if gate_step else "FM gate: not opened"
    best_info = f"best val rho={best_rho:.4f} @ ep{best_ep}" if vals else "val: in progress"
    title = f"{exp_name}  ({n_epochs} epochs completed)\n{gate_info}  |  {best_info}"

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    def vline(ax, use_step=True):
        if gate_step:
            x = gate_step if use_step else gate_ep
            ax.axvline(x, color=GATE_COLOR, lw=1.2, ls="--", alpha=0.8,
                       label=f"FM gate (s{gate_step})" if use_step else f"FM gate (ep{gate_ep})")

    # (0,0) Training loss
    ax = axes[0, 0]
    ax.plot(s_arr, rw_arr,  color="#3498db", alpha=0.25, lw=0.7)
    ax.plot(s_arr, smooth(rw_arr),  color="#3498db", lw=1.8, label="reward loss")
    ax.plot(s_arr, cd_arr,  color="#9b59b6", alpha=0.25, lw=0.7)
    ax.plot(s_arr, smooth(cd_arr),  color="#9b59b6", lw=1.8, label="CD loss")
    ax.plot(s_arr, reg_arr, color="#95a5a6", alpha=0.25, lw=0.7)
    ax.plot(s_arr, smooth(reg_arr), color="#95a5a6", lw=1.8, label="reg loss")
    vline(ax)
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (0,1) Val Spearman rho
    ax = axes[0, 1]
    ax.axhline(0.20,  color="gray",    ls=":",  lw=1,   alpha=0.6, label="rho=0.20 threshold")
    ax.axhline(0.239, color="#2980b9", ls=":",  lw=1.2, alpha=0.7, label="C avg test (0.239)")
    ax.axhline(0.279, color="#8e44ad", ls="--", lw=1.2, alpha=0.7, label="v1 best val (0.279)")
    if len(v_ep):
        ax.plot(v_ep, v_rho, color="#2c3e50", lw=1.5, alpha=0.6)
        ax.scatter(v_ep[v_pass],  v_rho[v_pass],  color=PASS_COLOR, s=55, zorder=5, label="PASS")
        ax.scatter(v_ep[~v_pass], v_rho[~v_pass], color=FAIL_COLOR, s=55, zorder=5,
                   marker="x", label="FAIL")
        if not np.isnan(best_rho):
            bi = int(np.argmax(v_rho))
            ax.scatter(v_ep[bi], v_rho[bi], color="gold", s=130, zorder=6,
                       edgecolors="black", lw=1, label=f"best ep{v_ep[bi]} ({v_rho[bi]:.4f})")
    vline(ax, use_step=False)
    ax.set_xlabel("epoch"); ax.set_ylabel("Spearman rho")
    ax.set_title("Val Spearman rho")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,0) Energy separation
    ax = axes[1, 0]
    ax.plot(s_arr, epos_arr, color=PASS_COLOR, lw=1.8, label="E+ (demo)")
    ax.fill_between(s_arr, epos_arr - epos_s, epos_arr + epos_s,
                    color=PASS_COLOR, alpha=ALPHA_BAND)
    ax.plot(s_arr, eneg_arr, color=FAIL_COLOR, lw=1.8, label="E- (negative)")
    ax.fill_between(s_arr, eneg_arr - eneg_s, eneg_arr + eneg_s,
                    color=FAIL_COLOR, alpha=ALPHA_BAND)
    vline(ax)
    ax.set_xlabel("step"); ax.set_ylabel("energy")
    ax.set_title("Energy Separation  (E+ vs E-,  +-1 sigma band)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) Val AUROC & ECE
    ax = axes[1, 1]
    ax2 = ax.twinx()
    if len(v_ep):
        ax.plot(v_ep, v_auc, color="#2980b9", lw=1.8, marker="o", ms=4, label="AUROC(E)")
        ax.axhline(0.688, color="#2980b9", ls=":", lw=1.2, alpha=0.7, label="C avg AUROC (0.688)")
        ax2.plot(v_ep, v_ece, color="#e67e22", lw=1.8, marker="s", ms=4, alpha=0.7, label="ECE")
    ax.set_xlabel("epoch"); ax.set_ylabel("AUROC(E)", color="#2980b9")
    ax2.set_ylabel("ECE", color="#e67e22")
    ax.set_title("Val AUROC(E) & ECE")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8)
    ax.grid(alpha=0.3)

    # (2,0) Reward grad norm
    ax = axes[2, 0]
    ax.semilogy(s_arr, grw_arr + 1e-6, color="#e74c3c", alpha=0.35, lw=0.7)
    ax.semilogy(s_arr, smooth(grw_arr + 1e-6, w=9), color="#e74c3c", lw=1.8,
                label="grad_rw (smoothed)")
    vline(ax)
    ax.set_xlabel("step"); ax.set_ylabel("grad norm (log)")
    ax.set_title("Reward Grad Norm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (2,1) Policy grad norm
    ax = axes[2, 1]
    ax.semilogy(s_arr, gpol_arr + 1e-6, color="#8e44ad", alpha=0.35, lw=0.7)
    ax.semilogy(s_arr, smooth(gpol_arr + 1e-6, w=9), color="#8e44ad", lw=1.8,
                label="grad_pol (smoothed)")
    vline(ax)
    ax.set_xlabel("step"); ax.set_ylabel("grad norm (log)")
    ax.set_title("Policy Grad Norm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (3,0) FM gate timeline & energy separation
    ax = axes[3, 0]
    ax.fill_between(s_arr, 0, 1, where=~fm_mask,
                    color="#bdc3c7", alpha=0.5,
                    transform=ax.get_xaxis_transform(), label="Phase 1: SGLD-only")
    ax.fill_between(s_arr, 0, 1, where=fm_mask,
                    color=GATE_COLOR, alpha=0.4,
                    transform=ax.get_xaxis_transform(), label="Phase 2: FM hybrid")
    ax.plot(s_arr, np.abs(epos_arr - eneg_arr), color="#2c3e50", lw=1.5,
            label="|E+ - E-| (separation)")
    vline(ax)
    ax.set_xlabel("step"); ax.set_ylabel("|E+ - E-|")
    ax.set_title("FM Gate Timeline & Energy Separation")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (3,1) Val rho vs previous experiments
    ax = axes[3, 1]
    ax.axhspan(-0.15, 0.0,  color=FAIL_COLOR,  alpha=0.05)
    ax.axhspan(0.0,   0.20, color="khaki",      alpha=0.08)
    ax.axhspan(0.20,  0.40, color=PASS_COLOR,   alpha=0.06)
    for label, val, col, ls in [
        ("C r1 best (0.236)", 0.2364, "#95a5a6", "--"),
        ("C r2 best (0.290)", 0.2902, "#95a5a6", "--"),
        ("C r3 best (0.295)", 0.2946, "#95a5a6", "--"),
        ("B test (0.208)",    0.2083, "#2980b9", ":"),
        ("v1 best (0.279)",   0.2791, "#8e44ad", "--"),
    ]:
        ax.axhline(val, ls=ls, lw=1, color=col, alpha=0.65, label=label)
    if len(v_ep):
        ax.plot(v_ep, v_rho, color="#2c3e50", lw=2)
        ax.scatter(v_ep[v_pass],  v_rho[v_pass],  color=PASS_COLOR, s=50, zorder=5)
        ax.scatter(v_ep[~v_pass], v_rho[~v_pass], color=FAIL_COLOR, s=50,
                   zorder=5, marker="x")
        if not np.isnan(best_rho):
            bi = int(np.argmax(v_rho))
            ax.scatter(v_ep[bi], v_rho[bi], color="gold", s=150, zorder=6,
                       edgecolors="black", lw=1.2)
            ax.text(v_ep[bi] + 0.3, v_rho[bi] + 0.005,
                    f"ep{v_ep[bi]}\n{v_rho[bi]:.4f}", fontsize=8, fontweight="bold")
    vline(ax, use_step=False)
    ax.set_xlabel("epoch"); ax.set_ylabel("Spearman rho")
    ax.set_title("Val rho vs. Previous Experiments")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved -> {save_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="실험 폴더 경로")
    args = parser.parse_args()
    generate_plot(Path(args.output_dir))
