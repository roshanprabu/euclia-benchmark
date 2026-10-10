"""Euclia agent (E1 all signals, E2 structured only). Needs: pip install anthropic; ANTHROPIC_API_KEY.
Outputs are cached per (condition, account) so reruns are free and reproducible.
Usage: python src/agent.py E1 [--limit N]"""
import json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-sonnet-5-5"
CACHE = ROOT / "results" / "cache"; CACHE.mkdir(parents=True, exist_ok=True)

SYSTEM = """You are Euclia, an AI Customer Success Manager for an SMB SaaS company.
Assess ONE account from the data given and return ONLY a JSON object with keys:
risk: "low"|"medium"|"high"
score: number 0-100 (churn/intervention urgency, used for ranking)
driver: "none"|"usage"|"billing"|"onboarding"|"support"|"relationship"|"competitive"
action: "no_action"|"reengagement_campaign"|"billing_followup"|"onboarding_playbook"|"upsell_outreach"|"human_escalation"
escalate_to_human: true if strategic, emotionally charged, competitive, relationship-loss, or sales-opportunity cases; false if a standard automated playbook is enough
explanation: 1-3 sentences citing the specific evidence.
Beware benign anomalies (seasonal dips, one-off late payments) and read free text carefully."""

def view(acc, structured_only):
    d = {k: v for k, v in acc.items() if not k.startswith("_")}
    if structured_only:
        d["support_tickets"] = [{"status": t["status"], "days_ago": t["days_ago"]} for t in d["support_tickets"]]
        d.pop("emails"); d.pop("csm_notes")
    return json.dumps(d)

def run(cond, limit=None):
    import anthropic
    client = anthropic.Anthropic()
    accounts = json.load(open(ROOT / "data" / "accounts_blind.json"))[:limit]
    out = {}
    for a in accounts:
        f = CACHE / f"{cond}_{a['account_id']}.json"
        if not f.exists():
            r = client.messages.create(model=MODEL, max_tokens=500, temperature=0, system=SYSTEM,
                messages=[{"role": "user", "content": view(a, cond == "E2")}])
            txt = r.content[0].text.strip().strip("`").removeprefix("json").strip()
            f.write_text(txt)
        out[a["account_id"]] = json.loads(f.read_text())
    (ROOT / "results" / f"pred_{cond}_euclia{'_structured' if cond=='E2' else ''}.json").write_text(json.dumps(out, indent=1))
    print("wrote", len(out), "predictions for", cond)

if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    run(sys.argv[1], lim)
