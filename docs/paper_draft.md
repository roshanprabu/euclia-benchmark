# Beyond Customer Success Automation: Designing and Evaluating a Human-in-the-Loop AI Agent for Proactive SaaS Retention

*DRAFT v0.2. All numbers come from `results/v2/` (final table: `results/v2/final_table.txt`). Claims are limited to a controlled synthetic benchmark. See §7 before citing any result.*

## Abstract
SMB SaaS companies cannot assign a customer success manager (CSM) to every account, and rule-based health scores miss risk that appears only in free text (a champion leaving, a competitor evaluation, escalating frustration). We study whether an LLM agent that reads structured and unstructured account signals can triage accounts, name the risk driver and decide when to escalate to a human. On a labeled synthetic benchmark (3 held-out seeds × 120 accounts, 9 archetypes, with independently authored free text and hard negatives) the agent far outperforms rule-based health scores on driver identification (0.89 vs 0.63 accuracy) and action selection (0.82 vs 0.59), while raising few false alarms (1% of low-risk accounts flagged high). Removing the free text collapses these gains, so reading text, not structured-data reasoning, drives them. A bare-prompt LLM detects risk about as well as our designed agent; the design's contribution is its escalation policy (escalation recall 0.82 vs 0.49). A supervised gradient-boosting model trained on 1,600 labeled accounts exceeds the agent on risk-level macro-F1 (0.75–0.79 vs 0.64) but produces many false alarms (53% of low-risk accounts when text is included) and poor driver attribution (0.60). Results are from synthetic data and do not show retention gains.

## 1. Introduction
- SMB SaaS firms cannot staff a CSM for every account; the long tail churns silently.
- Customer health is the central tracking construct in B2B customer success management, built from relationship quality, product usage and value realization (Hochstein et al., 2023). In practice it is a weighted rule score over structured data.
- Risk often appears only in text.
- **RQ.** Can an LLM account-level agent that fuses structured and unstructured signals identify, explain and route at-risk accounts better than rule-based health scoring, and where does it fail?
- **Contributions.** (1) A reproducible synthetic benchmark with ground-truth risk, driver, action and escalation labels, plus a documented procedure for finding and removing generator leakage; (2) a comparison of rule-based, supervised and zero-shot LLM approaches; (3) ablations isolating text and prompt design; (4) product implications: an escalation policy and MVP scope.
- **Gap (narrowed).** Prior work covers customer-success management and health (Hochstein 2020, 2023), text-assisted churn prediction (De Caigny et al., 2020; Chajia & Nfaoui, 2024), and LLM agents on general CRM and support tasks (CRMArena, JourneyBench). In a limited search we found no peer-reviewed or arXiv study of an LLM agent combining usage, billing, support and relationship text to produce risk, intervention and escalation decisions for B2B SaaS accounts (see docs/01_literature.md; absence is not proof, and commercial products claim similar features without published evaluation).

## 2. Related Work
[Write from docs/01_literature.md after full texts are read.] Strands: CS management and health; text and LLMs in churn prediction; LLM agents on CRM/support tasks; human-AI decision support and over-reliance (Bansal et al., 2021); synthetic-data validity.

## 3. Product Design
Monitor → Understand → Act → Escalate. Inputs: usage, billing, onboarding, tickets, emails, CSM notes. Output per account: risk, urgency score, primary driver, recommended action, escalation flag, evidence-citing explanation. Principle: automate standard playbooks (re-engagement, billing follow-up, onboarding); escalate strategic, relationship, competitive and expansion cases.

## 4. Method
### 4.1 Synthetic benchmark
Nine archetypes with hidden ground truth (risk, driver, action, escalate): healthy, expansion candidate, silent disengager, billing risk, onboarding stalled, frustrated power user, champion left, seasonal dip, competitor evaluation. 12 weeks of usage; billing, onboarding, 2–4 tickets, 2–3 emails, optional CSM notes. Three archetypes (champion left, competitor evaluation, frustrated power user) are intended to be visible only in text; seasonal dips and benign late payments mimic risk in structured data. **Free text was authored independently** of the generator by four subagents given only category descriptions (12 categories, 480 distinct strings), including hard negatives (a non-key employee leaving, an unrelated vendor evaluation, resolved irritated tickets). The text pool was split into disjoint halves: supervised baselines train on one half and all systems are evaluated on the other (zero shared strings).
### 4.2 Systems
B1 usage-only rule; B2 billing-only rule; B3 weighted rule-based health score; B4 gradient boosting on structured features; B5 gradient boosting on structured features + TF-IDF of all text (B4/B5 trained on 8 seeds × 200 labeled accounts from the same generator, so they enjoy in-distribution training); L0 LLM with a bare prompt (same output schema, no guidance); E2 Euclia agent, structured signals only; E1 Euclia agent, all signals (system prompt in `src/agent.py`: instructs on benign anomalies, escalation of strategic/relationship/competitive/expansion cases, evidence-citing explanations).
LLM runs were executed by independent Claude Code subagents (Sonnet), 30 accounts per subagent, each given only a label-free batch file and its instruction file. No labels existed on disk during the runs; labels were regenerated deterministically afterwards and checked against a SHA-256 hash of each evaluation set.
### 4.3 Metrics
Macro-F1 and high-risk recall/precision, driver accuracy, action match, escalation accuracy, per-archetype results, false-alarm rate on low-risk accounts; mean ± std over 3 seeds and a 1,000-sample bootstrap CI for macro-F1 on all 360 accounts.
### 4.4 Dataset iterations and leakage audit (disclosure)
Three generator versions were built. v1 (self-authored template text, 4% unique strings) leaked archetype through ticket/email counts (structured-only tree recovered the text-only archetypes at 64–85% recall). v2a (independent text) still leaked via onboarding-milestone counts, ticket counts/open-rate and a competitor-specific usage drift: a supervised model on structured fields alone recovered text-only archetypes at 59–70% risk accuracy. The generator was patched to equalize these distributions across archetypes and agent outputs on v2a were discarded (archived in `results/pilot_v2a_leaky/`). **The generator was therefore modified after seeing pilot results**, but only to remove structural leakage detected by the structured-only supervised probe; a final probe on fresh seeds recovered text-only archetypes at 19–35% risk accuracy. Final results use v2b with new seeds.

## 5. Results (3 seeds × 120 accounts)
### 5.1 Overall (`results/v2/final_table.txt`)
| System | Macro-F1 [95% CI] | High recall | High precision | Driver acc. | Action match | Escalation acc. |
|---|---|---|---|---|---|---|
| B1 usage-only | 0.33 [0.29,0.37] | 0.28 | 0.58 | 0.51 | 0.45 | 0.71 |
| B2 billing-only | 0.34 [0.30,0.38] | 0.20 | 1.00 | 0.56 | 0.51 | 0.71 |
| B3 health score | 0.56 [0.50,0.61] | 0.42 | 0.71 | 0.63 | 0.59 | 0.71 |
| B4 GB, structured | 0.79 [0.75,0.83] | 0.58 | 0.78 | 0.72 | 0.69 | 0.73 |
| B5 GB, structured + text | 0.75 [0.71,0.79] | 0.86 | 0.62 | 0.60 | 0.68 | 0.71 |
| L0 bare-prompt LLM | 0.63 [0.58,0.67] | 0.57 | 0.94 | 0.86 | 0.78 | 0.78 |
| E2 Euclia, structured only | 0.53 [0.47,0.59] | 0.42 | 0.83 | 0.59 | 0.49 | 0.60 |
| **E1 Euclia, all signals** | 0.64 [0.59,0.69] | 0.53 | 0.94 | **0.89** | **0.82** | **0.87** |

### 5.2 Detection of high-risk accounts (`results/v2/detection_by_archetype.txt`)
Share of true high-risk accounts flagged medium/high | correct driver among them | escalated: B3 0.58 | 0.50 | 0.00; B4 0.58 | 0.47 | 0.06; B5 0.86 | 0.58 | 0.38; L0 0.91 | 0.90 | 0.45; E2 0.56 | 0.50 | 0.24; **E1 0.92 | 0.87 | 0.51**. False alarms (low-risk accounts flagged high): B3 0.16, B4 0.16, **B5 0.53**, L0 0.00, E2 0.07, E1 0.01.
Text-only archetypes, share flagged not-low / driver correct: champion left: E1 0.71/0.50, L0 0.67/0.67, B5 0.62/0.29, B4 0.33/0.00; competitor evaluation: E1 0.84/0.77, L0 0.77/0.74, B5 0.65/0.16; frustrated power user: E1 1.00/0.90, L0 1.00/0.97, B5 0.93/0.23.
### 5.3 Findings
1. **Agent vs rule-based scoring.** E1 beats the best rule baseline (B3) on every metric reported: macro-F1 0.64 vs 0.56, driver accuracy 0.89 vs 0.63, action match 0.82 vs 0.59, escalation accuracy 0.87 vs 0.71, high recall 0.53 vs 0.42.
2. **Text is what matters.** E1 vs E2 (identical prompt): macro-F1 0.64 vs 0.53, driver 0.89 vs 0.59, action 0.82 vs 0.49, escalation 0.87 vs 0.60. Without text the LLM is no better than B3.
3. **Prompt design matters little for detection but matters for escalation.** E1 and L0 are indistinguishable on risk macro-F1 (overlapping CIs), driver accuracy (0.89 vs 0.86) and false alarms. Differences are in escalation: E1 escalates 82% of cases that should go to a human vs 49% for L0 (recall), and action match 0.82 vs 0.78 (small; 3 seeds, untested). The designed policy (when to hand off) is the added value.
4. **Supervised models are strong on risk-level F1 but weak on usefulness.** B4/B5 reach macro-F1 0.75–0.79 given labeled in-distribution data. B5 gets high recall (0.86) by over-flagging (53% of low-risk accounts) and names the wrong driver ~40% of the time; the LLMs read hard negatives correctly. B4 cannot see text-only archetypes (champion left 0.33, competitor 0.13, frustrated 0.17 risk accuracy). The LLM's advantage is zero-shot operation, driver attribution and false-alarm control, not top macro-F1.
5. **Failure mode: severity under-rating and rare cues.** E1 usually flags the right accounts but rates many as "medium" where labels say "high" (risk-level accuracy: champion left 0.00, competitor 0.32); champion departure is the hardest text cue (71% flagged, driver correct 50%).
### 5.4 Exploratory: forced-escalation rule (post hoc; `results/v2/escalation_rule_exploratory.txt`)
Escalating whenever the driver is relationship, competitive or support (or the action is upsell) raises L0's escalation recall from 0.49 to 0.86 and accuracy from 0.78 to 0.87; for E1 it changes little (recall 0.82 → 0.87, accuracy 0.867 → 0.875). The rule was derived after seeing results and should be re-tested on new seeds.

## 6. Product Implications
- **MVP: a read-only triage queue**, not an autonomous sender. Each row: risk, driver, evidence quotes, recommended action.
- **Autonomy matrix.** Auto-draft, human approves: billing follow-up, onboarding nudges, re-engagement for low-ARR accounts. Always route to a human: relationship, competitive, escalating-support and expansion drivers; enforce this as a deterministic rule outside the model (§5.4) rather than trusting the model's flag.
- **Do not rely on the model's severity.** Use driver + evidence for ranking and human review, and calibrate severity against real outcomes once available.
- **Cold-start path.** Zero-shot LLM triage first; once real churn labels accumulate, add a supervised model as a complementary score (B4/B5 show where it helps: structured risk) while keeping the LLM for text and explanation.
- **Production metrics:** high-risk recall on later-churned accounts, false-alarm rate on benign anomalies (seasonal, resolved tickets), escalation recall, CSM acceptance rate and time-to-triage.

## 7. Limitations
- **Synthetic, author-defined ground truth.** Labels encode the author's assumptions (e.g., "competitor evaluation = high risk"); a "medium" rating can be defensible. Results show relative behavior on a controlled benchmark, not real-world performance or retention.
- **Text authored by an LLM and evaluated by an LLM** of the same family (stylistic familiarity may favor the LLM systems). Text is short and cleaner than real communications; 480 distinct strings.
- **Supervised baselines trained on the same generator distribution** (favorable to them), but with disjoint text.
- **Small samples:** 3 seeds × 120 accounts; per-archetype cells are ~30–45 accounts; E1 vs L0 differences are not significance-tested.
- **Execution:** subagent runs (no sampling controls; not strictly reproducible), single LLM, single prompt per condition. Protocol deviations: one subagent listed the batch folder (filenames only, no labels); one batch wrote account IDs with an extra zero (normalized mechanically). `src/agent.py` provides a pinned-API path that has not been run.
- **Generator modified after pilot results** (leak removal, see §4.4); final probe shows residual but low structural signal.
- **No human evaluation.** Explanations may raise acceptance of AI advice irrespective of correctness (Bansal et al., 2021); usefulness to CSMs is untested.

## 8. Future Work
Human study with CSMs; real or anonymized production data and observed churn; pinned-model API replication with temperature 0 and more seeds; prompt/model sensitivity; severity calibration; hybrid LLM + supervised triage.

## 9. Conclusion
[Write after §2 is complete.]

## References
[Build only from sources actually read; see docs/01_literature.md.]
