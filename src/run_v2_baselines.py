import json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baselines import SYSTEMS
import trained_baselines as tb
ROOT = Path(__file__).resolve().parent.parent
train_files = sorted(glob.glob(str(ROOT / "data/v2/train/*/accounts.json")))
for d in sorted(glob.glob(str(ROOT / "data/v2/eval_s*"))):
    seed = Path(d).name.split("_s")[1]; out = ROOT / "results" / "v2" / f"s{seed}"; out.mkdir(parents=True, exist_ok=True)
    blind = json.load(open(Path(d) / "accounts_blind.json"))
    for n, fn in SYSTEMS.items():
        (out / f"pred_{n}.json").write_text(json.dumps({a["account_id"]: fn(a) for a in blind}, indent=1))
    tb.run(train_files, str(Path(d) / "accounts_blind.json"), str(out))
    print("seed", seed, "baselines done")
