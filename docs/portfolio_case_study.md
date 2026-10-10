# Euclia: Designing an AI Customer Success Agent (PM case study)

**Role:** Product manager and researcher (solo). **Artifact:** research paper, open benchmark, working evaluation harness.
**Repo:** https://github.com/roshanprabu/euclia-benchmark

## The problem
SMB SaaS companies can't afford a customer success manager (CSM) for every account. The long tail gets a rule-based "health score" built from logins, invoices and ticket counts, and it misses the signals CSMs act on: *the person who championed us just left*, *finance wants a side-by-side with a competitor before renewal*, *a power user has written three angry tickets*. Those live in emails and tickets, not in a usage chart.

## The bet
An AI agent that reads **all** of an account's signals (usage, billing, onboarding, tickets, emails, notes), explains its judgement with evidence, recommends an action, and **knows when to hand off to a human** could give long-tail accounts CSM-like attention.

## What I did
1. **Defined the job.** Monitor → Understand → Act → Escalate, with per-account output: risk, driver, action, escalate-or-not, evidence.
2. **Built a way to test it without customers.** I had no real account data, so I built a labeled synthetic benchmark: 9 account archetypes, 480 independently written messages, hard negatives (an intern leaving is not a champion leaving) and seasonal dips that look like churn but aren't.
3. **Audited my own benchmark.** A model trained only on structured fields could "detect" the text-only archetypes, which meant my generator leaked answers through ticket counts and onboarding patterns. I fixed it, threw away the results that depended on it, and re-ran on fresh seeds.
4. **Compared 8 systems:** three rule-based, two trained gradient-boosting models, a bare-prompt LLM, and my designed agent with and without text.

## What I found (360 accounts, 3 seeds)
- **The agent beat the rule-based health score on the things a CSM acts on:** naming the risk driver (0.89 vs 0.63), choosing the right action (0.82 vs 0.59), routing to a human correctly (0.87 vs 0.71), with ~1% false alarms.
- **It's the text, not the "AI reasoning."** Take the text away and the same agent drops to the level of the rule-based score.
- **My clever prompt wasn't what mattered for detection.** A bare-prompt LLM detected risk about as well. What the prompt design added was the **escalation policy**: it routed 82% of the accounts that should go to a human to a human, vs 49% for the bare prompt.
- **A trained model beat the agent on one metric and lost on usefulness.** Supervised models had higher risk-level F1, but the one that reads text flagged **53%** of healthy accounts and named the wrong driver ~40% of the time. The agent's edge is zero-shot operation (no churn labels needed), false-alarm control and driver explanation.
- **Where it fails:** it under-rates severity (calls "high" risks "medium") and is weakest at champion departure.

## Product decisions this drove
| Decision | Why |
|---|---|
| **MVP = read-only triage queue**, not an autonomous sender | Detection is good, severity calibration is not |
| **Auto-draft, human approves** for billing reminders, onboarding nudges, small-account re-engagement | Low-stakes, rule-like playbooks |
| **Always route relationship, competitive, escalating-support and expansion accounts to a human, via a hard rule outside the model** | A post-hoc rule lifted the bare LLM's escalation recall from 0.49 to 0.86 |
| **Rank by driver + evidence, not by the model's severity** | Severity under-rated |
| **Cold-start path:** LLM triage first, add a supervised score once real churn labels exist | Supervised needs labels a startup doesn't have yet |

**Metrics I'd track in production:** recall on accounts that later churned; false-alarm rate on seasonal dips and resolved tickets; escalation recall; CSM acceptance rate; time to triage.

## What I got wrong and what I learned
- **My first benchmark flattered the agent.** The numbers looked great until I tested whether the data could be solved without reading text. I now audit a benchmark before trusting any score on it.
- **I assumed the prompt was the product.** Ablating it showed the product was the *hand-off policy*.
- **Everything here is synthetic.** It shows relative behavior under controlled conditions, not retention lift. The next step is a pilot on real accounts with CSMs reviewing the queue.

## Resume bullet
*Designed and evaluated an LLM customer-success triage agent on a leakage-audited synthetic benchmark (8 systems, 360 accounts): improved risk-driver identification 0.63→0.89 and human-escalation routing 0.71→0.87 over a rule-based health score; found via ablation that gains came from reading text and from the escalation policy, and derived an autonomy/escalation framework for an MVP.*
