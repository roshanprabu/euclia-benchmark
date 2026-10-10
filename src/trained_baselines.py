"""Supervised baselines trained on other seeds' labeled data, tested on a held-out seed.
B4: gradient boosting on structured features. B5: B4 + TF-IDF of all free text (strong classical NLP baseline)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def feats(a):
    s = a["weekly_sessions"]; b = a["billing"]
    return [np.mean(s[:6]), np.mean(s[-4:]), np.mean(s[-4:]) / max(np.mean(s[:6]), 1), a["seats"],
            a["customer_since_months"], b["invoice_overdue_days"], b["failed_payments_90d"], int(b["downgrade_requested"]),
            sum(a["onboarding"].values()), sum(t["status"] == "open" for t in a["support_tickets"]),
            len(a["support_tickets"]), a["last_csm_contact_days"], len(a["emails"]), len(a["csm_notes"])]

def text(a):
    return " ".join([t["subject"] for t in a["support_tickets"]] + [e["text"] for e in a["emails"]] + [n["text"] for n in a["csm_notes"]])

def fit_predict(train, test, use_text):
    X = np.array([feats(a) for a in train]); Xt = np.array([feats(a) for a in test])
    if use_text:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, sublinear_tf=True)
        X = hstack([csr_matrix(X), v.fit_transform([text(a) for a in train])]).tocsr()
        Xt = hstack([csr_matrix(Xt), v.transform([text(a) for a in test])]).tocsr()
    out = {}
    for field in ("risk", "driver", "action", "escalate_to_human"):
        y = [a["_label"][field] for a in train]
        m = GradientBoostingClassifier(random_state=0, n_estimators=100, max_depth=3).fit(X, y)
        out[field] = (m.predict(Xt), m)
    rk = out["risk"][1]; P = rk.predict_proba(Xt); cl = list(rk.classes_)
    score = P[:, cl.index("high")] * 2 + (P[:, cl.index("medium")] if "medium" in cl else 0)
    return {a["account_id"]: {"risk": out["risk"][0][i], "driver": out["driver"][0][i], "action": out["action"][0][i],
            "escalate_to_human": bool(out["escalate_to_human"][0][i]), "score": float(score[i]), "explanation": "supervised"}
            for i, a in enumerate(test)}

def run(train_files, test_file, outdir):
    train = [a for f in train_files for a in json.load(open(f))]
    test = json.load(open(test_file))
    for name, ut in (("B4_gb_structured", False), ("B5_gb_structured_text", True)):
        preds = fit_predict(train, test, ut)
        Path(outdir, f"pred_{name}.json").write_text(json.dumps(preds, indent=1))

if __name__ == "__main__":
    run(sys.argv[3:], sys.argv[1], sys.argv[2])
