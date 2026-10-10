"""Score prediction files in results/pred_*.json against hidden labels."""
import json, glob
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, ndcg_score

ROOT = Path(__file__).resolve().parent.parent
RISK_NUM = {"low": 0, "medium": 1, "high": 2}

def score_system(preds, labels):
    ids = sorted(labels)
    y = [labels[i]["risk"] for i in ids]; p = [preds[i]["risk"] for i in ids]
    hi_t = np.array([r == "high" for r in y]); hi_p = np.array([r == "high" for r in p])
    tp = (hi_t & hi_p).sum()
    rel = np.array([[RISK_NUM[r] for r in y]])
    sc = np.array([[preds[i].get("score", RISK_NUM[preds[i]["risk"]]) + 0.01 * RISK_NUM[preds[i]["risk"]] for i in ids]])
    top10 = np.argsort(-sc[0])[:10]
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", labels=list(RISK_NUM)),
        "high_recall": tp / max(hi_t.sum(), 1),
        "high_precision": tp / max(hi_p.sum(), 1),
        "precision@10": float(hi_t[top10].mean()),
        "ndcg": float(ndcg_score(rel, sc)),
        "driver_acc": np.mean([preds[i]["driver"] == labels[i]["driver"] for i in ids]),
        "action_match": np.mean([preds[i]["action"] == labels[i]["action"] for i in ids]),
        "escalation_acc": np.mean([preds[i]["escalate_to_human"] == labels[i]["escalate_to_human"] for i in ids]),
    }

def per_archetype(preds, labels):
    out = {}
    for arch in sorted({l["archetype"] for l in labels.values()}):
        ids = [i for i in labels if labels[i]["archetype"] == arch]
        out[arch] = float(np.mean([preds[i]["risk"] == labels[i]["risk"] for i in ids]))
    return out

if __name__ == "__main__":
    accounts = json.load(open(ROOT / "data" / "accounts.json"))
    labels = {a["account_id"]: a["_label"] for a in accounts}
    rows, arch_rows = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results" / "pred_*.json"))):
        name = Path(f).stem[5:]
        preds = json.load(open(f))
        rows[name] = score_system(preds, labels); arch_rows[name] = per_archetype(preds, labels)
    cols = list(next(iter(rows.values())))
    print(f"{'system':18s}" + "".join(f"{c[:11]:>12s}" for c in cols))
    for n, r in rows.items(): print(f"{n:18s}" + "".join(f"{r[c]:12.3f}" for c in cols))
    print("\nRisk-level accuracy by archetype")
    archs = list(next(iter(arch_rows.values())))
    print(f"{'archetype':24s}" + "".join(f"{n[:10]:>12s}" for n in arch_rows))
    for a in archs: print(f"{a:24s}" + "".join(f"{arch_rows[n][a]:12.2f}" for n in arch_rows))
    (ROOT / "results" / "metrics.json").write_text(json.dumps({"overall": rows, "by_archetype": arch_rows}, indent=1, default=float))
