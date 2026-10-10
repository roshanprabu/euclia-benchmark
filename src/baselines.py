"""Rule-based baselines (B1 usage-only, B2 billing-only, B3 weighted health score).
They see only structured fields, never free text or labels."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTION_BY_DRIVER = {"usage": "reengagement_campaign", "billing": "billing_followup",
                    "onboarding": "onboarding_playbook", "support": "human_escalation",
                    "none": "no_action"}

def usage_drop(a):
    s = a["weekly_sessions"]
    early, late = sum(s[:6]) / 6, sum(s[-4:]) / 4
    return 0 if early == 0 else max(0.0, 1 - late / early)

def billing_flag(a):
    b = a["billing"]
    return b["invoice_overdue_days"] >= 14 or b["failed_payments_90d"] >= 1 or b["downgrade_requested"]

def onboarding_gap(a):
    return sum(not v for v in a["onboarding"].values())

def open_tickets(a):
    return sum(t["status"] == "open" for t in a["support_tickets"])

def pred(risk, driver, score=0.0):
    return {"risk": risk, "driver": driver, "action": ACTION_BY_DRIVER[driver],
            "escalate_to_human": driver == "support" and risk == "high",
            "score": score, "explanation": f"rule: {driver}"}

def b1(a):
    d = usage_drop(a)
    return pred("high", "usage", d) if d >= 0.30 else pred("low", "none", d)

def b2(a):
    return pred("high", "billing", 1.0) if billing_flag(a) else pred("low", "none", 0.0)

def b3(a):
    score = 0.0
    score += min(usage_drop(a) / 0.5, 1) * 40
    score += 30 if billing_flag(a) else 0
    score += onboarding_gap(a) * 6
    score += min(open_tickets(a), 3) * 7
    contrib = {"usage": min(usage_drop(a) / 0.5, 1) * 40, "billing": 30 if billing_flag(a) else 0,
               "onboarding": onboarding_gap(a) * 6, "support": min(open_tickets(a), 3) * 7}
    driver = max(contrib, key=contrib.get) if score > 0 else "none"
    risk = "high" if score >= 40 else "medium" if score >= 20 else "low"
    if risk == "low": driver = "none"
    return pred(risk, driver, score)

SYSTEMS = {"B1_usage_only": b1, "B2_billing_only": b2, "B3_health_score": b3}

if __name__ == "__main__":
    accounts = json.load(open(ROOT / "data" / "accounts_blind.json"))
    for name, fn in SYSTEMS.items():
        out = {a["account_id"]: fn(a) for a in accounts}
        (ROOT / "results" / f"pred_{name}.json").write_text(json.dumps(out, indent=1))
        print("wrote", name)
