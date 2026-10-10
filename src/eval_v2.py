"""Unseal labels (regenerate + hash-check), merge agent outputs, score all systems on v2 (3 seeds), bootstrap CIs."""
import json, subprocess, sys, hashlib, shutil
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import score_system, per_archetype

ROOT = Path(__file__).resolve().parent.parent
B = Path("/private/tmp/claude-501/-Users-roshanprabu-Documents-VSC-Projects-ai-pm/c9459700-1757-45d0-8f8f-07545f3bce2e/scratchpad/v2_batches")
POOL = B.parent / "textpool"
SEEDS = {101: 120, 102: 120, 103: 120}
RISK = {"low", "medium", "high"}; DRV = {"none","usage","billing","onboarding","support","relationship","competitive"}
ACT = {"no_action","reengagement_campaign","billing_followup","onboarding_playbook","upsell_outreach","human_escalation"}
hashes = json.load(open(ROOT / "data/v2/blind_hashes.json"))
SYS = ["B1_usage_only","B2_billing_only","B3_health_score","B4_gb_structured","B5_gb_structured_text","L0_bare_llm","E2_euclia_structured","E1_euclia"]
labels, preds = {}, {s: {} for s in SYS}
for seed, n in SEEDS.items():
    d = ROOT / "data/v2" / f"eval_s{seed}"
    subprocess.run([sys.executable, str(ROOT/"data/generate.py"), "--seed", str(seed), "--n", str(n), "--textpool", str(POOL),
                    "--poolpart", "b", "--out", str(d / "accounts.json")], check=True, capture_output=True)
    assert hashlib.sha256((d / "accounts_blind.json").read_bytes()).hexdigest() == hashes[str(seed)], "blind data changed!"
    for a in json.load(open(d / "accounts.json")): labels[f"s{seed}/{a['account_id']}"] = a["_label"]
    for s in SYS:
        if s.startswith(("B1","B2","B3","B4","B5")):
            p = json.load(open(ROOT / "results/v2" / f"s{seed}" / f"pred_{s}.json"))
        else:
            cond = {"L0_bare_llm": "L0", "E2_euclia_structured": "E2", "E1_euclia": "E1"}[s]; p = {}
            for f in sorted(B.glob(f"{cond}_s{seed}_out_*.json")): p.update(json.load(open(f)))
            fixed = {f"ACC-{int(k.split('-')[1]):04d}": v for k, v in p.items()}   # mechanical ID normalisation only
            nfix = sum(1 for k in p if k not in fixed); p = fixed
            if nfix: print(f"NOTE {s} seed {seed}: normalised {nfix} malformed account IDs (formatting only)")
            bad = [i for i, x in p.items() if x.get("risk") not in RISK or x.get("driver") not in DRV or x.get("action") not in ACT or not isinstance(x.get("escalate_to_human"), bool)]
            miss = n - len(p); 
            if bad or miss: print(f"WARN {s} seed {seed}: invalid={len(bad)} missing={miss}")
            for x in p.values(): x["score"] = float(x.get("score", 0))
            (ROOT / "results/v2" / f"s{seed}").mkdir(parents=True, exist_ok=True)
            (ROOT / "results/v2" / f"s{seed}" / f"pred_{s}.json").write_text(json.dumps(p, indent=1))
        for i, x in p.items(): preds[s][f"s{seed}/{i}"] = x
print("hash check passed: evaluation accounts identical to those the agents saw\n")
M = ["macro_f1","high_recall","high_precision","precision@10","driver_acc","action_match","escalation_acc"]
per_seed = {s: [score_system({k: preds[s][k] for k in labels if k.startswith(f"s{sd}/")}, {k: labels[k] for k in labels if k.startswith(f"s{sd}/")}) for sd in SEEDS] for s in SYS}
rng = np.random.default_rng(0); keys = sorted(labels); y = np.array([labels[k]["risk"] for k in keys])
def boot(s):
    p = np.array([preds[s][k]["risk"] for k in keys]); v = []
    for _ in range(1000):
        i = rng.integers(0, len(keys), len(keys)); v.append(f1_score(y[i], p[i], average="macro", labels=["low","medium","high"]))
    return np.percentile(v, [2.5, 97.5])
res = {}
print(f"{'system':24s}" + "".join(f"{m[:12]:>14s}" for m in M) + "   macroF1 95% CI")
for s in SYS:
    row = {m: (float(np.mean([r[m] for r in per_seed[s]])), float(np.std([r[m] for r in per_seed[s]]))) for m in M}
    lo, hi = boot(s); row["macro_f1_ci"] = [float(lo), float(hi)]; res[s] = row
    print(f"{s:24s}" + "".join(f"{row[m][0]:8.3f}±{row[m][1]:.2f}" for m in M) + f"   [{lo:.3f},{hi:.3f}]")
arch = {s: per_archetype(preds[s], labels) for s in SYS}
print("\nRisk-level accuracy by archetype (pooled, 360 accounts)")
print(f"{'archetype':24s}" + "".join(f"{s[:9]:>10s}" for s in SYS))
for a in sorted(next(iter(arch.values()))): print(f"{a:24s}" + "".join(f"{arch[s][a]:10.2f}" for s in SYS))
(ROOT / "results/v2/metrics.json").write_text(json.dumps({"overall": res, "by_archetype": arch}, indent=1))
for seed in SEEDS: (ROOT / "data/v2" / f"eval_s{seed}" / "accounts.json").exists()
