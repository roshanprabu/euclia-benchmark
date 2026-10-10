"""Synthetic SMB SaaS account generator for the Euclia study.

Each account is drawn from a hidden archetype that determines its ground-truth
risk level, primary risk driver, correct intervention, and whether a human CSM
should handle it. Some archetypes are only detectable through unstructured text
(tickets, emails, notes); others are benign anomalies designed to trigger
false alarms in rule-based systems.

Usage: python generate.py --n 200 --seed 42 --noise 0.15
"""
import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np

WEEKS = 12
TODAY = date(2026, 10, 1)
POOL = None  # independent text pool, set via --textpool

# name: (weight, risk, driver, action, escalate)
ARCHETYPES = {
    "healthy":               (0.30, "low",    "none",        "no_action",            False),
    "expansion_candidate":   (0.08, "low",    "none",        "upsell_outreach",      True),
    "silent_disengager":     (0.12, "high",   "usage",       "reengagement_campaign", False),
    "billing_risk":          (0.10, "high",   "billing",     "billing_followup",     False),
    "onboarding_stalled":    (0.10, "medium", "onboarding",  "onboarding_playbook",  False),
    "frustrated_power_user": (0.08, "high",   "support",     "human_escalation",     True),
    "champion_left":         (0.08, "high",   "relationship", "human_escalation",    True),
    "seasonal_dip":          (0.07, "low",    "none",        "no_action",            False),
    "competitor_eval":       (0.07, "high",   "competitive", "human_escalation",     True),
}

INDUSTRIES = ["E-commerce", "Agency", "Healthcare clinic", "Accounting firm",
              "Real estate", "Education", "Logistics", "Hospitality", "Legal"]
NAME_A = ["Blue", "North", "Bright", "Iron", "Maple", "Summit", "Clear", "Pine",
          "Harbor", "Silver", "Cedar", "Atlas", "Nova", "River", "Oak", "Lumen"]
NAME_B = ["Labs", "Works", "Co", "Group", "Studio", "Partners", "Health",
          "Logistics", "Digital", "Supply", "Legal", "Realty", "Academy"]
MILESTONES = ["account_setup", "first_integration", "team_invited",
              "first_workflow", "first_report"]

NEUTRAL_TICKETS = [
    "How do I export a report to CSV?",
    "Can we add another admin to the workspace?",
    "Question about configuring notification settings.",
    "Where can I find the API key?",
    "Is there a way to bulk-import contacts?",
]
NEUTRAL_EMAILS = [
    "Thanks for the update, looks good.",
    "Appreciate the help last week, team is getting value.",
    "Quick question on the new feature, will reply in thread.",
]


def pick_archetype(rng):
    names = list(ARCHETYPES)
    weights = [ARCHETYPES[n][0] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def usage_series(arch, base, rng, noise):
    t = np.arange(WEEKS)
    if arch == "silent_disengager":
        start = rng.randint(3, 7)
        curve = np.where(t < start, 1.0, np.exp(-0.25 * (t - start)))
    elif arch == "onboarding_stalled":
        curve = np.clip(0.25 + 0.02 * t, 0, 1)
    elif arch == "seasonal_dip":
        curve = np.ones(WEEKS)
        curve[-4:-1] = 0.45  # temporary dip, recovering in final week
        curve[-1] = 0.7
    elif arch == "expansion_candidate":
        curve = 1.0 + 0.06 * t
    else:
        curve = np.ones(WEEKS)
        if arch in ("healthy", "champion_left", "frustrated_power_user", "competitor_eval") and rng.random() < 0.15:
            curve = np.where(t < 9, 1.0, 0.8)  # same mild drift rate in all four archetypes
    vals = base * curve * (1 + np.random.normal(0, noise, WEEKS))
    return [max(0, int(round(v))) for v in vals]


def billing(arch, mrr, rng):
    overdue_days, failed = 0, 0
    if arch == "billing_risk":
        overdue_days, failed = rng.choice([21, 35, 48]), rng.randint(1, 3)
    elif rng.random() < 0.08:  # benign late payment noise
        overdue_days = rng.choice([3, 7, 12])
    renewal = TODAY + timedelta(days=rng.randint(10, 300))
    return {"mrr_usd": mrr, "plan": rng.choice(["Starter", "Growth", "Pro"]),
            "invoice_overdue_days": overdue_days, "failed_payments_90d": failed,
            "renewal_date": renewal.isoformat(),
            "downgrade_requested": arch == "billing_risk" and rng.random() < 0.4}


def onboarding(arch, rng):
    if arch == "onboarding_stalled":
        done = rng.randint(1, 2)
    else:  # identical for every other archetype, so milestone count carries no archetype signal
        done = rng.choices([5, 4, 3, 2], weights=[0.65, 0.2, 0.1, 0.05])[0]
    return {m: i < done for i, m in enumerate(MILESTONES)}


def _p(rng, *opts):
    return rng.choice(opts)

def angry_ticket(rng):
    if POOL: return rng.choice(POOL['frustrated'])
    return " ".join([
        _p(rng, "This is the third time the", "Again the", "Once more the", "For the second week the"),
        _p(rng, "sync", "report export", "data import", "dashboard", "integration", "scheduled job") + " "
        + _p(rng, "has broken.", "failed.", "is returning wrong numbers.", "stopped working."),
        _p(rng, "Our whole team is blocked.", "We're losing patience.", "Nobody has followed up.",
           "This is costing us real money.", "We pay for this and still hit bugs."),
        _p(rng, "Need an answer this week.", "Please escalate.", "", "Please call me.")]).strip()

def mild_ticket(rng, closed=True):  # hard negative: irritated tone, resolved, no risk
    if POOL: return rng.choice(POOL['mild_resolved_ticket'])
    return " ".join([
        _p(rng, "Slightly annoyed that the", "Minor issue: the", "FYI the", "Small glitch, the"),
        _p(rng, "export", "login page", "report", "filter", "invoice PDF") + " "
        + _p(rng, "was slow earlier.", "showed an error once.", "looked off.", "needed a refresh."),
        _p(rng, "Works now, thanks.", "Resolved after retry.", "Not urgent.", "Fixed itself.")])

def neutral_ticket(rng):
    if POOL: return rng.choice(POOL['neutral_ticket'])
    return " ".join([_p(rng, "How do I", "Can we", "Is there a way to", "Where can I"),
        _p(rng, "export a report?", "add an admin?", "change notification settings?",
           "find the API key?", "bulk-import contacts?", "rename a workspace?", "merge two projects?")])

def tickets(arch, rng):
    n = rng.randint(2, 4)
    items = []
    for _ in range(n):
        kind = rng.random()
        subj = mild_ticket(rng) if kind < 0.25 else neutral_ticket(rng)
        items.append({"subject": subj, "status": "open" if rng.random() < 0.25 else "closed",
                      "days_ago": rng.randint(1, 80)})
    special = []
    if arch == "frustrated_power_user":
        special = [{"subject": angry_ticket(rng), "status": "open" if rng.random() < 0.3 else "closed",
                    "days_ago": rng.randint(1, 20)} for _ in range(2)]
    elif arch == "onboarding_stalled":
        special = [{"subject": _p(rng, "We're not sure how to connect our existing tools, is there a guide?",
                    "Stuck on the first workflow setup, which step comes next?",
                    "Our team hasn't been able to get started beyond account creation."),
                    "status": "open", "days_ago": rng.randint(10, 30)}]
    elif arch == "billing_risk":
        special = [{"subject": _p(rng, "Can we discuss our invoice? Budget is tight this quarter.",
                    "Our card was declined, finance is looking into it.",
                    "Need to understand charges before we pay."),
                    "status": "open", "days_ago": rng.randint(3, 15)}]
    if POOL and arch in ('onboarding_stalled', 'billing_risk'):
        for sp in special:
            sp['subject'] = rng.choice(POOL['onboarding_stalled' if arch == 'onboarding_stalled' else 'billing'])
    items = items[: max(0, n - len(special))] + special   # replace, don't add: keeps counts matched
    rng.shuffle(items)
    return items

EMAIL_NEUTRAL = ["Thanks for the update, looks good.", "Appreciate the help last week, team is getting value.",
    "Quick question on the new feature, will reply in thread.", "Sounds good, let's catch up next month.",
    "Can you resend the onboarding deck?", "Team loves the new dashboard."]
EMAIL_HARD_NEG = [  # superficially similar to risk text but benign
    "Our intern finished her placement; no change for us, same team otherwise.",
    "We're evaluating accounting software for finance, unrelated to this tool.",
    "A colleague is on parental leave for a few months; I'm covering and fully up to speed.",
    "Just comparing notes with a peer company on how they use the tool.",
    "Reorg on our side, but the admin and day-to-day owner are unchanged."]

def comms(arch, rng):
    """Emails and CSM notes: the only place some risks appear."""
    emails = []
    for _ in range(rng.randint(2, 3)):
        txt = ((rng.choice(POOL['hard_negative_email']) if POOL else rng.choice(EMAIL_HARD_NEG)) if rng.random() < 0.18
               else (rng.choice(POOL['neutral_email']) if POOL else rng.choice(EMAIL_NEUTRAL)))
        emails.append({"from": "customer", "text": txt, "days_ago": rng.randint(3, 80)})
    notes = [{"author": "csm", "days_ago": rng.randint(5, 60),
              "text": _p(rng, "Check-in call went fine.", "Customer asked about roadmap.", "Sent quarterly summary.")}
             ] if rng.random() < 0.4 else []
    special = None
    if arch == "champion_left":
        special = _p(rng, "Hi, just letting you know {n} has left the company. I'm taking over for now but still getting up to speed.",
                     "Our ops lead {n}, who set this up, moved on last week. Not sure who owns this tool going forward.",
                     "{n} was our main contact and is no longer with us. We haven't decided who takes over.").format(
                     n=_p(rng, "Sarah", "Mike", "Priya", "Daniel", "Alex", "Maria"))
    elif arch == "competitor_eval":
        special = _p(rng, "Before renewal we're reviewing a few options in the market. Can you send pricing for annual?",
                     "Our leadership asked us to compare a couple of alternatives. What's on your roadmap for Q1?",
                     "We got a demo from another vendor and want to understand how you differ.",
                     "Finance wants a side-by-side with other tools before we renew.")
    elif arch == "expansion_candidate":
        special = _p(rng, "We're hiring 10 more people next quarter, is there a volume discount for more seats?",
                     "Planning to roll this out to a second department, what would that cost?")
    elif arch == "seasonal_dip":
        notes.append({"author": "csm", "days_ago": rng.randint(20, 40), "text": _p(rng,
            "Customer mentioned their busy season ends in late September; expect lower activity for a few weeks.",
            "Team is on annual shutdown for ~3 weeks, usage will dip. Not a concern.")})
    if POOL and special:
        special = rng.choice(POOL[{'champion_left': 'champion_left', 'competitor_eval': 'competitor_eval', 'expansion_candidate': 'expansion'}[arch]])
    if POOL:
        for nt in notes:
            nt['text'] = rng.choice(POOL['seasonal_note'] if (arch == 'seasonal_dip' and nt is notes[-1]) else POOL['routine_note'])
    if special:
        emails[rng.randrange(len(emails))] = {"from": "customer", "text": special, "days_ago": rng.randint(3, 14)}
    if arch == "silent_disengager":  # tends to go quiet, but healthy accounts sometimes do too
        emails = [e for e in emails if e["days_ago"] > 45] or emails[:1]
    elif rng.random() < 0.15:
        emails = [e for e in emails if e["days_ago"] > 45] or emails[:1]
    return emails, notes


def make_account(i, rng, noise):
    arch = pick_archetype(rng)
    _, risk, driver, action, escalate = ARCHETYPES[arch]
    seats = rng.randint(3, 40)
    base = seats * rng.uniform(2.5, 5)  # weekly active sessions
    mrr = seats * rng.choice([15, 25, 49])
    emails, notes = comms(arch, rng)
    return {
        "account_id": f"ACC-{i:04d}",
        "name": f"{rng.choice(NAME_A)} {rng.choice(NAME_B)}",
        "industry": rng.choice(INDUSTRIES),
        "seats": seats,
        "customer_since_months": rng.randint(2, 36) if arch != "onboarding_stalled" else rng.randint(1, 3),
        "weekly_sessions": usage_series(arch, base, rng, noise),
        "billing": billing(arch, mrr, rng),
        "onboarding": onboarding(arch, rng),
        "support_tickets": tickets(arch, rng),
        "emails": emails,
        "csm_notes": notes,
        "last_csm_contact_days": rng.randint(5, 120),
        "_label": {"archetype": arch, "risk": risk, "driver": driver,
                   "action": action, "escalate_to_human": escalate},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--textpool", default=None, help="dir of pool*.json with independently written text")
    p.add_argument("--poolpart", choices=["a", "b"], default=None, help="use disjoint half of each pool (a=even idx, b=odd idx)")
    p.add_argument("--out", default=str(Path(__file__).parent / "accounts.json"))
    a = p.parse_args()

    global POOL
    if a.textpool:
        POOL = {}
        for f in sorted(Path(a.textpool).glob('pool*.json')): POOL.update(json.load(open(f)))
        if a.poolpart:
            POOL = {k: v[0::2] if a.poolpart == 'a' else v[1::2] for k, v in POOL.items()}
    rng = random.Random(a.seed)
    np.random.seed(a.seed)
    accounts = [make_account(i, rng, a.noise) for i in range(1, a.n + 1)]

    Path(a.out).write_text(json.dumps(accounts, indent=2))
    # Label-free copy for systems under test, so labels can't leak.
    blind = [{k: v for k, v in acc.items() if k != "_label"} for acc in accounts]
    Path(a.out).with_name("accounts_blind.json").write_text(json.dumps(blind, indent=2))

    counts = {}
    for acc in accounts:
        counts[acc["_label"]["archetype"]] = counts.get(acc["_label"]["archetype"], 0) + 1
    print(f"Wrote {len(accounts)} accounts (seed={a.seed}, noise={a.noise})")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
