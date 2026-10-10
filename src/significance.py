"""Paired significance tests over the 360 pooled v2 accounts.
Paired bootstrap (10,000 resamples) for metric differences; exact McNemar (binomial on discordant pairs) for per-account correctness."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (101, 102, 103)
def load():
    L, P = {}, {}
    for s in SEEDS:
        for a in json.load(open(ROOT / f"data/v2/eval_s{s}/accounts.json")): L[f"s{s}/{a['account_id']}"] = a["_label"]
        for f in (ROOT / f"results/v2/s{s}").glob("pred_*.json"):
            for k, v in json.load(open(f)).items():
                P.setdefault(f.stem[5:], {})[f"s{s}/ACC-{int(k.split('-')[1]):04d}"] = v
    return L, P
L, P = load(); K = sorted(L)
y = {f: np.array([L[k][f] for k in K]) for f in ("risk", "driver", "action", "escalate_to_human")}
def arr(sysn, f): return np.array([P[sysn][k][f] for k in K])
def correct(sysn, f): return arr(sysn, f) == y[f]
def mf1(yt, yp): return f1_score(yt, yp, average="macro", labels=["low", "medium", "high"])
rng = np.random.default_rng(0); IDX = rng.integers(0, len(K), (10000, len(K)))
out = []
PAIRS = [("E1_euclia", "L0_bare_llm"), ("E1_euclia", "B3_health_score"), ("E1_euclia", "E2_euclia_structured"), ("E1_euclia", "B4_gb_structured"), ("E1_euclia", "B5_gb_structured_text")]
out.append(f"{'comparison':40s}{'metric':16s}{'A':>7s}{'B':>7s}{'diff':>8s}   95% CI (paired bootstrap)   McNemar p")
for a, b in PAIRS:
    for name, f in (("driver_acc", "driver"), ("action_match", "action"), ("escalation_acc", "escalate_to_human")):
        ca, cb = correct(a, f), correct(b, f); d = ca.astype(float) - cb
        boots = d[IDX].mean(1); lo, hi = np.percentile(boots, [2.5, 97.5])
        n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        out.append(f"{a+' vs '+b:40s}{name:16s}{ca.mean():7.3f}{cb.mean():7.3f}{d.mean():+8.3f}   [{lo:+.3f}, {hi:+.3f}]          {p:.2g}")
    ra, rb = arr(a, "risk"), arr(b, "risk")
    diffs = np.array([mf1(y['risk'][i], ra[i]) - mf1(y['risk'][i], rb[i]) for i in IDX[:2000]])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out.append(f"{a+' vs '+b:40s}{'risk_macroF1':16s}{mf1(y['risk'], ra):7.3f}{mf1(y['risk'], rb):7.3f}{mf1(y['risk'], ra)-mf1(y['risk'], rb):+8.3f}   [{lo:+.3f}, {hi:+.3f}]          (bootstrap only, 2,000)")
txt = "\n".join(out); print(txt); (ROOT / "results/v2/significance.txt").write_text(txt + "\n")
