"""Sanity/validity checks on the synthetic dataset (resemblance, diversity, leakage)."""
import json, collections
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
accs = json.load(open(ROOT/"data"/"accounts.json"))
y = [a["_label"]["archetype"] for a in accs]

print("Class balance (risk):", dict(collections.Counter(a["_label"]["risk"] for a in accs)))

# Diversity of free text
texts = [t["subject"] for a in accs for t in a["support_tickets"]] + [e["text"] for a in accs for e in a["emails"]] + [n["text"] for a in accs for n in a["csm_notes"]]
print(f"Free-text diversity: {len(set(texts))} unique / {len(texts)} total ({len(set(texts))/len(texts):.1%})")

# Leakage check: can STRUCTURED features alone recover the archetype?
def feats(a):
    s = a["weekly_sessions"]; b = a["billing"]
    return [np.mean(s[:6]), np.mean(s[-4:]), np.mean(s[-4:])/max(np.mean(s[:6]),1), a["seats"],
            a["customer_since_months"], b["invoice_overdue_days"], b["failed_payments_90d"],
            int(b["downgrade_requested"]), sum(a["onboarding"].values()),
            sum(t["status"]=="open" for t in a["support_tickets"]), len(a["support_tickets"]),
            a["last_csm_contact_days"], len(a["emails"]), len(a["csm_notes"])]
X = np.array([feats(a) for a in accs])
pred = cross_val_predict(DecisionTreeClassifier(max_depth=6, random_state=0), X, y, cv=5)
print("\nStructured-only decision tree (5-fold), recall by archetype:")
for arch in sorted(set(y)):
    idx = [i for i, v in enumerate(y) if v == arch]
    print(f"  {arch:24s} {np.mean([pred[i]==arch for i in idx]):.2f}")
print("\nInterpretation: archetypes with low recall are only separable via text, which is the intended design.")
print("Archetypes with HIGH recall from structured fields alone are 'easy'. Report this honestly in Limitations.")
