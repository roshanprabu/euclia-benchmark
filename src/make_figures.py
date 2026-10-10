"""Figures for the manuscript. Colour = system family (validated palette); values labelled on marks."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
M = json.load(open(ROOT / "results/v2/metrics.json"))["overall"]
FAM = {"Rule-based": "#2a78d6", "Supervised (trained)": "#eb6834", "LLM (zero-shot)": "#1baf7a"}
SYS = [("B1_usage_only", "B1 Usage-only rule", "Rule-based"), ("B2_billing_only", "B2 Billing-only rule", "Rule-based"),
       ("B3_health_score", "B3 Health score", "Rule-based"), ("B4_gb_structured", "B4 GB, structured", "Supervised (trained)"),
       ("B5_gb_structured_text", "B5 GB, structured+text", "Supervised (trained)"), ("L0_bare_llm", "L0 Bare-prompt LLM", "LLM (zero-shot)"),
       ("E2_euclia_structured", "E2 Euclia, no text", "LLM (zero-shot)"), ("E1_euclia", "E1 Euclia, all signals", "LLM (zero-shot)")]
INK, GRID, MUTED = "#0b0b0b", "#e3e2de", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: four metrics, systems on y, colour by family
panels = [("macro_f1", "Risk macro-F1"), ("driver_acc", "Driver accuracy"), ("action_match", "Action match"), ("escalation_acc", "Escalation accuracy")]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
ys = np.arange(len(SYS))[::-1]
for ax, (key, title) in zip(axes, panels):
    for y, (k, lab, fam) in zip(ys, SYS):
        v = M[k][key][0]
        ax.barh(y, v, height=0.55, color=FAM[fam], zorder=3)
        if key == "macro_f1":
            lo, hi = M[k]["macro_f1_ci"]; ax.plot([lo, hi], [y, y], color=INK, lw=1, zorder=4)
        ax.text(min(v + 0.02, 0.93) if key != "macro_f1" else max(v, M[k]["macro_f1_ci"][1]) + 0.02, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1.05); ax.set_title(title, fontsize=9.5, loc="left", color=INK); ax.set_xticks([0, .5, 1]); ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(length=0); ax.spines["left"].set_visible(False)
axes[0].set_yticks(ys); axes[0].set_yticklabels([s[1] for s in SYS])
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM.values()]
fig.legend(handles, FAM.keys(), loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(ROOT / "paper/fig1_overall.png", dpi=300, facecolor="white"); plt.close(fig)

# Figure 2: detection vs false alarms (computed from saved predictions)
labels, P = {}, {}
for seed in (101, 102, 103):
    for a in json.load(open(ROOT / f"data/v2/eval_s{seed}/accounts.json")): labels[f"s{seed}/{a['account_id']}"] = a["_label"]
    for k, _, _ in SYS:
        for i, v in json.load(open(ROOT / f"results/v2/s{seed}/pred_{k}.json")).items(): P.setdefault(k, {})[f"s{seed}/ACC-{int(i.split('-')[1]):04d}"] = v
hi = [k for k, l in labels.items() if l["risk"] == "high"]; lo = [k for k, l in labels.items() if l["risk"] == "low"]
MARK = {"Rule-based": "s", "Supervised (trained)": "^", "LLM (zero-shot)": "o"}
fig, ax = plt.subplots(figsize=(6.2, 4.4))
off = {"B1_usage_only": (6, -10), "B2_billing_only": (6, 6), "B3_health_score": (6, -4), "B4_gb_structured": (6, -12), "B5_gb_structured_text": (-8, 8), "L0_bare_llm": (8, 8), "E2_euclia_structured": (6, -4), "E1_euclia": (8, -14)}
pts = {}
for k, lab, fam in SYS:
    x = np.mean([P[k][i]["risk"] == "high" for i in lo]); y = np.mean([P[k][i]["risk"] != "low" for i in hi]); pts[k] = (x, y)
    ax.scatter(x, y, s=70, color=FAM[fam], marker=MARK[fam], edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(lab.split(" ")[0], (x, y), textcoords="offset points", xytext=off[k], fontsize=8.5, color=INK, ha="left" if off[k][0] > 0 else "right")
ax.set_xlabel("False-alarm rate (low-risk accounts flagged high)"); ax.set_ylabel("Detection (high-risk accounts flagged medium/high)")
ax.set_xlim(-0.03, 0.62); ax.set_ylim(0, 1.05); ax.yaxis.grid(True, color=GRID, zorder=0); ax.xaxis.grid(True, color=GRID, zorder=0); ax.tick_params(length=0)
ax.legend([plt.Line2D([], [], marker=MARK[f], color=FAM[f], linestyle="", markersize=7) for f in FAM], FAM.keys(), frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(ROOT / "paper/fig2_detection.png", dpi=300, facecolor="white"); plt.close(fig)
json.dump({k: [float(v[0]), float(v[1])] for k, v in pts.items()}, open(ROOT / "paper/fig2_points.json", "w"), indent=1)
print({k: [round(float(a), 2) for a in v] for k, v in pts.items()})
