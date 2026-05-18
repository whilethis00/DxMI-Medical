"""
Regenerate RESULT.md for a finished experiment directory.

This script intentionally avoids importing train.py so it can run quickly after
test evaluation in wrapper scripts.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import yaml


VAL_LOG_RE = re.compile(
    r"(?:VAL\s+ep(?P<ep1>\d+)|\[epoch\s+(?P<ep2>\d+)\]\s+val:)"
    r".*?(?:Spearman\s+)?(?:ρ|rho)=(?P<rho>[+-]?\d+\.\d+)"
    r"\s+\(p=(?P<p>[\d.]+)(?:,\s*(?P<status1>PASS|FAIL))?\)"
    r".*?(?:\[(?P<status2>PASS|FAIL)|)"
    r".*?(?:AUROC\(E\)|AUROC)=(?P<auroc>[\d.]+)"
    r".*?ECE=(?P<ece>[\d.]+)"
)

TEST_LOG_RE = re.compile(
    r"\[TEST\]\s+(?P<name>[^|]+?)\s+\|\s+ckpt=(?P<ckpt>[^|]+?)\s+\|\s+N=(?P<n>\d+)"
    r".*?Spearman\s+(?:ρ|rho)=(?P<rho>[+-]?\d+\.\d+)"
    r"\s+\(p=(?P<p>[\d.]+),\s*(?P<status>PASS|FAIL)\)"
    r"(?:.*?AUROC\(-E\)=(?P<auroc_neg>[\d.]+))?"
    r".*?AUROC\(E\)=(?P<auroc>[\d.]+)"
    r".*?ECE=(?P<ece>[\d.]+)"
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_val_logs(log_file: Path) -> list[dict]:
    vals = []
    if not log_file.exists():
        return vals

    for line in log_file.read_text().splitlines():
        m = VAL_LOG_RE.search(line)
        if not m:
            continue
        n_match = re.search(r"\bN=(\d+)", line)
        rho = float(m.group("rho"))
        p_value = float(m.group("p"))
        vals.append({
            "epoch": int(m.group("ep1") or m.group("ep2")),
            "rho": rho,
            "p": p_value,
            "status": "PASS" if rho > 0 and p_value < 0.05 else "FAIL",
            "auroc": float(m.group("auroc")),
            "ece": float(m.group("ece")),
            "n": int(n_match.group(1)) if n_match else None,
            "best_mark": "BEST" in line,
        })
    return vals


def parse_test_log(test_file: Path) -> dict | None:
    if not test_file.exists():
        return None
    for line in test_file.read_text().splitlines():
        m = TEST_LOG_RE.search(line)
        if not m:
            continue
        return {
            "name": m.group("name").strip(),
            "ckpt": m.group("ckpt").strip(),
            "n": int(m.group("n")),
            "rho": float(m.group("rho")),
            "p": float(m.group("p")),
            "status": "PASS" if float(m.group("rho")) > 0 and float(m.group("p")) < 0.05 else "FAIL",
            "auroc_neg": float(m.group("auroc_neg")) if m.group("auroc_neg") else None,
            "auroc": float(m.group("auroc")),
            "ece": float(m.group("ece")),
        }
    return None


def write_result_md(cfg: dict, config_path: str, exp_dir: Path):
    exp_info = cfg.get("experiment", {})
    training = cfg.get("training", {})
    vals = parse_val_logs(exp_dir / "train.log")
    test = parse_test_log(exp_dir / "test.log")
    last_val = vals[-1] if vals else None
    best_val = max(vals, key=lambda x: x["rho"]) if vals else None

    result_md = exp_dir / "RESULT.md"
    with open(result_md, "w") as f:
        f.write(f"# {exp_info.get('name', 'exp')} 실험 결과\n\n")
        f.write(f"날짜: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"config: `{config_path}`\n\n")
        f.write("---\n\n")
        f.write("## 개요\n\n")
        f.write(f"{exp_info.get('description', '')}\n\n")
        f.write("## 가설\n\n")
        f.write("*(작성 필요)*\n\n")
        f.write("## 실험 세팅\n\n")
        f.write("| 항목 | 값 |\n|------|----|\n")
        f.write(f"| epochs | {training.get('epochs', '?')} |\n")
        f.write(f"| l2_reg | {training.get('l2_reg', '?')} |\n")
        f.write(f"| reward_cd_weight | {training.get('reward_cd_weight', 'N/A')} |\n")
        f.write(f"| reward_cd_temp | {training.get('reward_cd_temp', 'N/A')} |\n")
        f.write(f"| fm_gate_sep_std_threshold | {training.get('fm_gate_sep_std_threshold', '?')} |\n\n")
        f.write("## 결과\n\n")

        if vals:
            f.write("| split | epoch/ckpt | Spearman rho | p-value | AUROC(E) | ECE | N | status |\n")
            f.write("|-------|------------|--------------|---------|----------|-----|---|--------|\n")
            if best_val:
                f.write(
                    f"| best val | ep{best_val['epoch']:02d} | {best_val['rho']:.4f} | "
                    f"{best_val['p']:.4f} | {best_val['auroc']:.4f} | {best_val['ece']:.4f} | "
                    f"{best_val['n'] or ''} | {best_val['status']} |\n"
                )
            if last_val:
                f.write(
                    f"| final val | ep{last_val['epoch']:02d} | {last_val['rho']:.4f} | "
                    f"{last_val['p']:.4f} | {last_val['auroc']:.4f} | {last_val['ece']:.4f} | "
                    f"{last_val['n'] or ''} | {last_val['status']} |\n"
                )
            if test:
                f.write(
                    f"| test | {test['ckpt']} | {test['rho']:.4f} | {test['p']:.4f} | "
                    f"{test['auroc']:.4f} | {test['ece']:.4f} | {test['n']} | {test['status']} |\n"
                )
            f.write("\n")
            f.write("### Val 전체 로그\n\n")
            f.write("| epoch | Spearman rho | p-value | AUROC(E) | ECE | N | status | best |\n")
            f.write("|-------|--------------|---------|----------|-----|---|--------|------|\n")
            for v in vals:
                f.write(
                    f"| ep{v['epoch']:02d} | {v['rho']:.4f} | {v['p']:.4f} | "
                    f"{v['auroc']:.4f} | {v['ece']:.4f} | {v['n'] or ''} | "
                    f"{v['status']} | {'yes' if v['best_mark'] else ''} |\n"
                )
        else:
            f.write("val 결과 없음\n")
        f.write("\n")

        if last_val:
            f.write(
                f"**최종 val**: ep{last_val['epoch']:02d}, rho={last_val['rho']:.4f}, "
                f"p={last_val['p']:.4f}, AUROC(E)={last_val['auroc']:.4f}, "
                f"ECE={last_val['ece']:.4f}, {last_val['status']}\n\n"
            )
        else:
            f.write("**최종 val**: `val 결과 없음`\n\n")
        if test:
            f.write(
                f"**Test**: ckpt={test['ckpt']}, rho={test['rho']:.4f}, "
                f"p={test['p']:.4f}, AUROC(E)={test['auroc']:.4f}, "
                f"ECE={test['ece']:.4f}, {test['status']}\n\n"
            )
        f.write("## 가설 달성 여부\n\n")
        f.write("*(작성 필요)*\n\n")
        f.write("## 인사이트\n\n")
        f.write("*(작성 필요)*\n\n")
        f.write("## 다음 계획\n\n")
        f.write("*(작성 필요)*\n")

    print(f"[INFO] RESULT.md 생성: {result_md}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    write_result_md(load_config(args.config), args.config, Path(args.output_dir))


if __name__ == "__main__":
    main()
