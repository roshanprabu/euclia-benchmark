# Phase 1 — Literature Review Outline

> **Update (revision 2):** the novelty framing below was narrowed after a second search pass. CRMArena (NAACL 2025) already benchmarks LLM agents on CRM tasks, so **do not claim that no LLM-agent CRM research exists.** See "Revised gap" and "Candidate reference pool" at the end.

**Rule:** only cite what you have opened and read. Items marked ✅ were found via web search (abstract level only, read the full paper before citing). Items marked 🔍 are search directions, not citations.

## 1. Customer health scoring and SaaS churn
**Gap to establish:** health scores are widely used in practice but thinly studied academically.
- ✅ Carezzato, A. A. M. (2019) — FGV (Brazil) master's dissertation, 40,635 customers and 396 candidate variables, so health-score-based SaaS churn prediction is NOT unexplored. building a churn model from a Customer Health Score for an ERP vendor. Portuguese. Notes SaaS health-score metrics are under-researched. https://eaesp.fgv.br/producao-intelectual/customer-experience-e-customer-health-score-clientes-saas-analise
- 🔍 Google Scholar: "subscription churn prediction machine learning", "B2B SaaS churn", "customer success management" (Information Systems / Marketing journals), "customer engagement metrics SaaS".
- ⚠️ Vendor blogs (Statisfy, BuildBetter, etc.) make statistics like "health scores predict churn in under 35% of CS teams" with no traceable source. Use only as evidence of practice, never as data.

## 2. NLP / LLMs for churn and customer text
**Gap:** most work uses embeddings or sentiment as features for a classifier; little on LLM agents reasoning over mixed structured and unstructured account data.
- ✅ 2024 *Future Internet* paper: LLM embeddings + logistic regression for telecom churn. https://doaj.org/article/bb4be8eb98d94042aea17b5a17374f06 (verify title/authors)
- ✅ NCI (Ireland) 2025 master's thesis: RAG-augmented LLaMA sentiment features into CatBoost, on Amazon reviews. Thesis, not peer-reviewed. https://norma.ncirl.ie/9430/
- 🔍 "support ticket escalation prediction" (a vendor cites a University of Victoria / IBM study, unverified), "LLM customer support agents evaluation".

## 3. Human-AI decision making and explanations (relevant to the human-in-the-loop framing and Future Work)
- ✅ Bansal et al., CHI 2021, "Does the Whole Exceed its Parts? The Effect of AI Explanations on Complementary Team Performance." Explanations raised acceptance of AI advice whether or not it was correct, rather than improving team accuracy. https://arxiv.org/abs/2006.14779
- Why it matters: it argues for treating "explanations improve decisions" as a hypothesis, which justifies deferring the human study to Future Work and for not asserting it.
- 🔍 "appropriate reliance on AI", "automation bias", "human-in-the-loop AI escalation policies", "selective prediction / learning to defer".

## 4. Evaluating with synthetic data (methodology and limitations)
- ✅ SynAE (arXiv 2605.22564) — framework for measuring validity, fidelity and diversity of synthetic data for agent evaluation. https://arxiv.org/pdf/2605.22564
- ✅ ESDAE — ICLR 2026, same problem; reports no single metric captures synthetic data quality. https://iclr.cc/virtual/2026/10018721
- ✅ "Quality Matters" (EMNLP 2024) — high error rates in synthetic tool-use benchmarks. (Find the full citation before using.)
- Use: motivates your limitations section and justifies multi-seed and noise-sweep robustness checks. Consider reporting basic validity/diversity checks on your own dataset.

## 5. Agent autonomy and escalation design (for the product-implications section)
- 🔍 "levels of autonomy AI agents", "human oversight of AI agents", "AI agent customer service deployment case studies".

## How the review argues
1. Health scores are standard practice but under-studied; rule-based versions rely on structured data.
2. Text and relationship signals predict churn in practice, but LLM approaches mostly use text as classifier features, not reasoning over the whole account.
3. Human-AI work shows explanations can inflate trust, so treat them as a hypothesis, test only machine-side performance here, and design explicit escalation gates.
4. Synthetic evaluation is increasingly common but easy to overclaim; state validity limits openly.

## Your to-do (this phase)
- Open and read each ✅ item; fix full citations.
- Run the 🔍 searches in Google Scholar, ACM DL and AIS eLibrary; aim for 20–30 total sources.
- Build a literature matrix (source | method | data | finding | relevance).

---

## Revised gap (use this, not "nothing exists")
Prior work separately covers (a) B2B customer-success management and customer health, (b) text-assisted churn prediction, and (c) LLM-agent performance on general CRM or support tasks. The narrower, candidate gap: **how an account-level agent combines longitudinal usage, billing, support and relationship signals into evidence-grounded health assessments and prioritized interventions, and how reliably it does so under noisy or incomplete data.** This is a candidate, not a confirmed novelty claim. Before asserting it, read the related-work sections of CRMArena and JourneyBench and check for account-level next-best-action work.

## Verification status
Opened at abstract/listing level by me (still read full texts before citing):
- Hochstein, Voorhees, Pratt, Rangarajan, Nagel, Mehrotra (2023), *Int. J. of Research in Marketing* 40(4), 912–932, doi:10.1016/j.ijresmar.2023.09.002. Customer health as a formative metric (relationship quality, product usage, value realization). This is your anchor for the signal categories in the dataset.
- Huang et al. (2025), CRMArena, NAACL long.194, pp. 3830–3850. https://aclanthology.org/2025.naacl-long.194 — **the arXiv v2 and NAACL versions report different success rates (<40%/<55% vs <58%/<65%). Quote only the version you cite.**
- Bansal et al. (2021), CHI. Carezzato (2019). Chajia & Nfaoui (2024), *Future Internet*.

Not independently verified (came from a pasted shortlist): everything else in the pool below, including all items dated 2026. Check each exists and the details match before citing.
- SynAE and ESDAE have the same authors and overlapping abstracts. Treat as one work until publication records show otherwise.
- Theses (Anderson 2019, Suzuki 2024, NCI 2025) are weak evidence. Cite sparingly and label as theses.

## Candidate reference pool (from shortlist, unverified unless noted above)
**Customer success / churn / CRM:** Hochstein et al. 2020 (*J. Service Research*, CSM research agenda) · De Caigny et al. 2020 (text in churn models, CNN) · AI-in-CRM literature review (212 articles) · Anderson 2019 (thesis) · Suzuki 2024 (case study) · Ledro et al. 2026.
**Human–AI decision support:** Bansal 2021 · Wang & Yin 2022 · Naiseh et al. 2023 · Vered et al. 2023 · Westphal et al. 2023.
**Agent evaluation:** CRMArena 2025 · JourneyBench 2026 · AgentBench 2023 · GAIA 2024 · WebArena (ICLR 2024) · Huang et al. 2024 (tool utilization) · ToolSandbox 2025 · ACEBench 2025 · Yehudai et al. 2026 (survey).
**Synthetic data validity:** SynAE/ESDAE · "Can I trust my fake data?" 2024 · health-domain synthetic tabular evaluation frameworks (2).

## Next actions
1. Read full texts: Hochstein 2023, Hochstein 2020, Carezzato, Chajia & Nfaoui, CRMArena, JourneyBench.
2. Mine their reference lists for direct customer-success and next-best-action work instead of adding loosely related AI papers.
3. Then finalize the four-step review argument and the novelty sentence.
4. Cut the pool to the 20–30 you actually read. Drop generic benchmarks (GAIA, WebArena) unless you use them.

---

## Revision 3: novelty check by search agent (2026-10-10)
Full report: scratchpad `lit_check.md` (copy key facts here). Caveat: related-work text was read through summaries; JourneyBench PDF text could not be parsed.
- **CRMArena** (Huang et al., NAACL 2025; arXiv 2411.02305): 9 service/analyst/manager CRM tasks (case routing, policy violation, KB QA, etc.). Does not cover churn/health, usage/billing time series, emails/notes, CS interventions or escalation.
- **JourneyBench** = "Beyond IVR: Benchmarking Customer Support LLM Agents for Business-Adherence" (Balaji et al.; **EACL 2026 Industry track**, arXiv 2601.00596; not ACL main). Support-policy adherence; no B2B account-level risk seen. Related work inferred from citation keys only.
- **CARRE** (arXiv 2609.09766, reported by the search agent, unverified by me): telecom tabular churn + counterfactual action retrieval + LLM explanation. Closest on "risk → action → explanation"; no B2B SaaS, support/email signals or escalation. Open it before citing.
- Commercial products (Gainsight, ChurnZero, Totango, Planhat, Darwin) claim fused health + next-best-action with human owners; no published evaluation. Prior art for the idea, not the study.
- **No peer-reviewed or arXiv paper found** evaluating an LLM agent over usage + billing + support + email/notes for B2B SaaS risk, interventions and escalation. ~5 standard-mode queries; absence is not proof.
- Citations confirmed at metadata level: Hochstein et al. 2023 (IJRM 40(4):912–932); De Caigny et al. 2020 (**Int. J. of Forecasting** 36(4):1563–1578, not a marketing venue); Chajia & Nfaoui 2024 (Future Internet 16(12):453). Partly confirmed: Wang & Yin 2022 (ACM TiiS 12(4), doi 10.1145/3519266; title/pages unconfirmed); Vered et al. 2023 (*Artificial Intelligence*; volume/DOI unconfirmed).
