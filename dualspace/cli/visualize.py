from __future__ import annotations
import json, click
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


@click.group("visualize")
def visualize():
    """Plot evaluation figures (coverage curves, Pareto)."""
    pass


@visualize.command("coverage")
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False), required=True,
              help="e.g., outputs/cifar10_mvp")
def plot_coverage(run_dir: str):
    """Plot coverage vs α on TEST (per-class + macro if available)."""
    mdir = Path(run_dir) / "metrics"
    data = json.loads((mdir / "coverage_curve_test.json").read_text(encoding="utf-8"))

    # try per-class+macro first; else pooled
    if "coverage_test_per_class" in data:
        cov_pc = data["coverage_test_per_class"]
        macro = data.get("coverage_test_macro", {})
        alphas = [float(a) for a in next(iter(cov_pc.values())).keys()]
        # plot per-class faint, macro bold, diagonal
        plt.figure(figsize=(4, 3))
        for k, d in cov_pc.items():
            ys = [d[f"{a:.3f}"] for a in alphas]
            plt.plot(alphas, ys, color="C0", alpha=0.25, linewidth=1)
        if macro:
            ys = [macro[f"{a:.3f}"] for a in alphas]
            plt.plot(alphas, ys, color="C1", linewidth=2, label="macro")
        plt.plot(alphas, alphas, "k--", linewidth=1, label="target")
        plt.xlabel("target α"); plt.ylabel("empirical coverage (test)")
        plt.ylim(0, 1.0); plt.xlim(min(alphas), max(alphas))
        plt.legend(frameon=False)
        plt.tight_layout()
        out = mdir / "coverage_curve_test.png"
        plt.savefig(out, dpi=200); plt.close()
        click.echo(f"[visualize.coverage] wrote {out}")
    else:
        pooled = data["coverage_test_pooled"]
        alphas = [float(a) for a in pooled.keys()]
        ys = [pooled[f"{a:.3f}"] for a in alphas]
        plt.figure(figsize=(4, 3))
        plt.plot(alphas, ys, color="C1", linewidth=2, label="pooled")
        plt.plot(alphas, alphas, "k--", linewidth=1, label="target")
        plt.xlabel("target α"); plt.ylabel("empirical coverage (test)")
        plt.ylim(0, 1.0); plt.xlim(min(alphas), max(alphas))
        plt.legend(frameon=False)
        plt.tight_layout()
        out = mdir / "coverage_curve_test.png"
        plt.savefig(out, dpi=200); plt.close()
        click.echo(f"[visualize.coverage] wrote {out}")


@visualize.command("pareto")
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False), required=True)
def plot_pareto(run_dir: str):
    """Plot Pareto: region size (log-det cov in φ) vs coverage on test."""
    mdir = Path(run_dir) / "metrics"
    data = json.loads((mdir / "pareto_points.json").read_text(encoding="utf-8"))
    alphas = data["alpha_grid"]
    per_class = data.get("per_class", True)
    points = data["points"]

    plt.figure(figsize=(5, 4))
    for a in alphas:
        xs, ys = [], []
        for k, seq in points.items():
            rec = next(r for r in seq if abs(r["alpha"] - a) < 1e-9)
            xs.append(rec["region_logdetcov"])
            ys.append(rec["coverage_test"])
        plt.scatter(xs, ys, s=16, alpha=0.8, label=f"α={a:.2f}")
    plt.xlabel("region size proxy: logdet(cov_φ) ↑ (more diffuse)")
    plt.ylabel("coverage on test ↑")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    out = mdir / "pareto_logdetcov_vs_coverage.png"
    plt.legend(frameon=False, fontsize=8)
    plt.savefig(out, dpi=200); plt.close()
    click.echo(f"[visualize.pareto] wrote {out}")
