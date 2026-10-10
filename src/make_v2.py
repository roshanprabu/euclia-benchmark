"""Build the v2 evaluation: independent text pool, 3 held-out eval seeds, N training seeds for supervised baselines.
Labels for eval seeds are DELETED from disk before agents run and regenerated (deterministically) for scoring."""
import json, subprocess, sys, hashlib, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import agent

ROOT = Path(__file__).resolve().parent.parent
SCR = Path("/private/tmp/claude-501/-Users-roshanprabu-Documents-VSC-Projects-ai-pm/c9459700-1757-45d0-8f8f-07545f3bce2e/scratchpad")
POOL = SCR / "textpool"
EVAL = {101: 120, 102: 120, 103: 120}
TRAIN = {s: 200 for s in range(201, 209)}
BATCH = 30
BARE = ("You are an assistant for a SaaS company. You are given one customer account as JSON. Assess churn risk and suggest what to do. "
        "Return ONLY a JSON object with keys: risk (low|medium|high), score (0-100), driver (none|usage|billing|onboarding|support|relationship|competitive), "
        "action (no_action|reengagement_campaign|billing_followup|onboarding_playbook|upsell_outreach|human_escalation), escalate_to_human (true|false), explanation (1-2 sentences).")

def gen(seed, n, out, part):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"), "--seed", str(seed), "--n", str(n),
                    "--textpool", str(POOL), "--poolpart", part, "--out", str(out)], check=True, capture_output=True)

def main():
    V2 = ROOT / "data" / "v2"; B = SCR / "v2_batches"; B.mkdir(parents=True, exist_ok=True)
    (B / "SYSTEM_E.txt").write_text(agent.SYSTEM); (B / "SYSTEM_BARE.txt").write_text(BARE)
    hashes = {}
    for seed, n in TRAIN.items(): gen(seed, n, V2 / "train" / f"s{seed}" / "accounts.json", "a")
    for seed, n in EVAL.items():
        d = V2 / f"eval_s{seed}"; gen(seed, n, d / "accounts.json", "b")
        blind = json.load(open(d / "accounts_blind.json"))
        hashes[seed] = hashlib.sha256((d / "accounts_blind.json").read_bytes()).hexdigest()
        for cond, structured in (("E1", False), ("E2", True), ("L0", False)):
            for k in range(0, len(blind), BATCH):
                batch = [json.loads(agent.view(a, structured)) for a in blind[k:k + BATCH]]
                (B / f"{cond}_s{seed}_in_{k // BATCH}.json").write_text(json.dumps(batch))
        (d / "accounts.json").unlink()   # seal labels: not on disk while agents run
    (V2 / "blind_hashes.json").write_text(json.dumps(hashes))
    print("batches:", len(list(B.glob("*_in_*.json"))), "| labels sealed for eval seeds")

if __name__ == "__main__":
    main()
