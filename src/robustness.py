"""Sweep seeds x noise levels; report baseline metrics as mean +/- std."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from baselines import SYSTEMS
from evaluate import score_system

ROOT = Path(__file__).resolve().parent.parent
SEEDS, NOISES = range(1, 11), [0.05, 0.15, 0.30, 0.50]
METRICS = ["macro_f1", "high_recall", "high_precision", "action_match"]

def main():
    res = {n: {} for n in SYSTEMS}
    with tempfile.TemporaryDirectory() as tmp:
        for noise in NOISES:
            runs = {n: [] for n in SYSTEMS}
            for seed in SEEDS:
                out = Path(tmp) / f"a_{seed}_{noise}.json"
                subprocess.run([sys.executable, str(ROOT/"data"/"generate.py"), "--seed", str(seed),
                                "--noise", str(noise), "--out", str(out)], check=True, capture_output=True)
                accs = json.load(open(out)); labels = {a["account_id"]: a["_label"] for a in accs}
                blind = [{k: v for k, v in a.items() if k != "_label"} for a in accs]
                for n, fn in SYSTEMS.items():
                    runs[n].append(score_system({a["account_id"]: fn(a) for a in blind}, labels))
            for n in SYSTEMS:
                res[n][noise] = {m: (float(np.mean([r[m] for r in runs[n]])), float(np.std([r[m] for r in runs[n]]))) for m in METRICS}
    (ROOT/"results"/"robustness.json").write_text(json.dumps(res, indent=1))
    for m in METRICS:
        print(f"\n{m} (mean±std over {len(SEEDS)} seeds)")
        print(f"{'system':18s}" + "".join(f"{'noise='+str(x):>16s}" for x in NOISES))
        for n in SYSTEMS:
            print(f"{n:18s}" + "".join(f"{res[n][x][m][0]:8.3f}±{res[n][x][m][1]:.3f}" for x in NOISES))

if __name__ == "__main__":
    main()
