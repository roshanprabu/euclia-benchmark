# Phase 0 — Research Scope

## Working title
Beyond Customer Success Automation: Designing and Evaluating a Human-in-the-Loop AI Agent for Proactive SaaS Retention

## Research question
Can an LLM-based customer success agent that fuses fragmented account signals (usage, billing, support, onboarding, communications) identify and prioritize at-risk SMB SaaS accounts — and propose appropriate interventions — more accurately than a conventional rule-based health score?

## Design constraint
No human participants. All evaluation is run on a **synthetic, labeled account dataset** with known ground truth (generated in Phase 2). Human-subject hypotheses (decision quality, time savings) are moved to Future Work.

## Hypotheses
- **H1 — Signal fusion.** A multi-signal approach detects at-risk accounts more accurately (macro-F1, recall on high-risk) than single-signal baselines (usage-only, billing-only).
- **H2 — Unstructured context.** An LLM agent that reads unstructured signals (ticket text, emails, CSM notes) outperforms a rule-based health score on archetypes whose risk is only visible in text (e.g., champion departure, frustrated power user) and produces fewer false alarms on benign anomalies (e.g., seasonal dips).
- **H3 — Actionability.** The agent's recommended interventions match the ground-truth appropriate action more often than the rule-based system's fixed playbook mapping, and its explanations cite the correct risk driver.

## Conditions compared
| ID | System | Inputs |
|----|--------|--------|
| B1 | Usage-only rule | Usage trend |
| B2 | Billing-only rule | Billing status |
| B3 | Rule-based health score (industry-style weighted) | All structured signals |
| E1 | Euclia agent (LLM) | All structured + unstructured signals |
| E2 | Euclia ablation | Structured signals only (isolates the value of text) |

## Metrics
- Risk classification: accuracy, macro-F1, high-risk recall/precision
- Prioritization: Precision@10, NDCG over accounts ranked by risk
- Driver attribution: % of explanations naming the correct primary driver
- Action match: % of recommended actions matching the ground-truth action
- Escalation calibration: % correctly routed to human vs. automated handling
- Per-archetype breakdown (where each system fails)

## Out of scope
- Real churn prediction or claims of retention lift
- Real CRM integrations or sending actual messages
- Human-subject studies (Future Work)

## Limitations to state up front
Synthetic data encodes the author's assumptions about churn patterns; results show *relative* capability under controlled conditions, not real-world performance. Mitigated by: grounding archetypes in published CS literature and practitioner sources, releasing the generator, and testing robustness across noise levels and random seeds.

## Deliverables
1. Synthetic dataset + generator (reproducible, seeded)
2. Baselines + Euclia agent prototype
3. Evaluation harness and results
4. Product implications: MVP scope and autonomy/escalation framework
5. Paper draft + portfolio case study
