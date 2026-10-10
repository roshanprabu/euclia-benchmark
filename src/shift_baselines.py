"""Distribution-shift test for supervised baselines: train on accounts with a different archetype mix
(uniform instead of the default skew) and higher usage noise (0.30 vs 0.15), then test on the unchanged v2 evaluation seeds."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import trained_baselines as tb
from evaluate import score_system
ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data/textpool"; OUT = ROOT / "data/v2/train_shift"
files = []
for seed in range(401, 409):
    f = OUT / f"s{seed}/accounts.json"; f.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "data/generate.py"), "--seed", str(seed), "--n", "200", "--textpool", str(POOL),
                    "--poolpart", "a", "--mix", "uniform", "--noise", "0.30", "--out", str(f)], check=True, capture_output=True)
    files.append(str(f))
rows = {}
for name in ("B4_gb_structured", "B5_gb_structured_text"):
    rows[name] = {"in-distribution": [], "shifted": []}
for seed in (101, 102, 103):
    d = ROOT / f"data/v2/eval_s{seed}"; L = {a["account_id"]: a["_label"] for a in json.load(open(d / "accounts.json"))}
    out = ROOT / f"results/v2/shift/s{seed}"; out.mkdir(parents=True, exist_ok=True)
    tb.run(files, str(d / "accounts_blind.json"), str(out))
    for name in rows:
        rows[name]["shifted"].append(score_system(json.load(open(out / f"pred_{name}.json")), L))
        rows[name]["in-distribution"].append(score_system(json.load(open(ROOT / f"results/v2/s{seed}/pred_{name}.json")), L))
import numpy as np
M = ["macro_f1", "high_recall", "high_precision", "driver_acc", "action_match", "escalation_acc"]
lines = ["Supervised baselines: trained in-distribution vs trained on shifted data (uniform archetype mix, noise 0.30); mean over 3 eval seeds", f"{'system':24s}{'training':18s}" + "".join(f"{m[:12]:>13s}" for m in M)]
for name, r in rows.items():
    for cond, lst in r.items(): lines.append(f"{name:24s}{cond:18s}" + "".join(f"{np.mean([x[m] for x in lst]):13.3f}" for m in M))
txt = "\n".join(lines); print(txt); (ROOT / "results/v2/shift_baselines.txt").write_text(txt + "\n")
