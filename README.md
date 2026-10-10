# Euclia benchmark

Synthetic benchmark and evaluation code for the study *Beyond Customer Success Automation: Designing and Evaluating a Human-in-the-Loop AI Agent for Proactive SaaS Retention*.

**All data are synthetic.** No real customer or personal data were used. Results show relative behavior on a controlled benchmark, not real-world accuracy or retention gains.

## Contents
| Path | What it is |
|---|---|
| `data/generate.py` | Seeded generator for 9 account archetypes with hidden ground-truth labels |
| `data/textpool/` | 480 independently authored free-text strings (12 categories) used by the generator |
| `data/v2/` | Final evaluation (3 seeds × 120 accounts: `eval_s*/`, labeled + label-free) and training accounts for supervised baselines (`train/`) |
| `src/` | Baselines, supervised baselines, agent prompt (`agent.py`), evaluation, robustness, validity and figure scripts |
| `results/v2/` | Final tables, per-system predictions, raw LLM-agent outputs (`agent_raw/`), exploratory analyses |
| `results/pilot_v2a_leaky/` | Discarded pilot run on a version of the generator that still leaked structure (kept for transparency) |
| `docs/` | Scope, literature notes, paper draft |
| `paper/` | Figures |

Top-level `data/accounts*.json` and `results/*.json` are the earliest pilot (version 1, template text) and are superseded.

## Reproducing
1. `python src/make_v2.py` builds sealed evaluation seeds and training seeds from `data/textpool/` (edit the scratchpad path constants in the script).
2. LLM conditions (L0 bare prompt, E1 all signals, E2 structured only) were run by independent Claude Code subagents on label-free batch files. Instructions: `results/v2/agent_raw/SYSTEM_*.txt`. Raw outputs: `results/v2/agent_raw/*_out_*.json`.
3. `python src/run_v2_baselines.py` then `python src/eval_v2.py` regenerates labels deterministically, verifies SHA-256 hashes (`data/v2/blind_hashes.json`), and scores all systems.
4. `src/agent.py` provides a pinned-API, temperature-0 replication path (`pip install anthropic`, set `ANTHROPIC_API_KEY`). It has not been run yet.

Python dependencies: numpy, scikit-learn, scipy, matplotlib.

## Known limitations
Synthetic, author-defined labels; text authored by Claude instances and evaluated with Claude models; small samples; subagent execution without sampling controls. See `docs/paper_draft.md` section 7.

## License
MIT (`LICENSE`), covering the code, the synthetic data and the text pool.

## Citation
Prabu, R. (2026). *Euclia benchmark* [Data set and code]. GitHub. https://github.com/roshanprabu/euclia-benchmark
