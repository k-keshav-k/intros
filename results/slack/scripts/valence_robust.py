"""Self vs observer letter-valence correlation: single reports and equal-count averages (mean over 200 subsample draws)."""
import json, sys
import numpy as np
from regexintro import grammars, diagnose, data
from boolintro.boolfn import corr

DRAWS = 200

def boot(x, rng):
    bs = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    return np.percentile(bs, [2.5, 97.5])

def run(tag, base, L):
    rng = np.random.default_rng(0)
    spec = data.load_spec(f"{base}/L{L}/spec.json"); rule = {c["id"]: c["rule"] for c in spec["characters"]}
    single, eq = [], []
    for line in open(f"{base}/L{L}/eval/step_004000_reports.jsonl"):
        r = json.loads(line)
        vt = diagnose.valence(grammars.language(rule[r["id"]]))
        selfp = [x for x in (grammars.parse_report(t) for t in r["reports"]) if x is not None]
        obsp = [x for x in (grammars.parse_report(t) for t in (r["observer_reports"] or [])) if x is not None]
        if not selfp or not obsp: continue
        vs = [diagnose.valence(grammars.language(x)) for x in selfp]; vo = [diagnose.valence(grammars.language(x)) for x in obsp]
        with np.errstate(all="ignore"):
            single.append(np.nanmean([corr(v, vt) for v in vs]) - np.nanmean([corr(v, vt) for v in vo]))
            k = min(len(vs), len(vo)); d = []
            for _ in range(DRAWS):
                a = corr(np.mean([vs[i] for i in rng.choice(len(vs), k, replace=False)], 0), vt)
                b = corr(np.mean([vo[i] for i in rng.choice(len(vo), k, replace=False)], 0), vt)
                if not (np.isnan(a) or np.isnan(b)): d.append(a - b)
        if d: eq.append(np.mean(d))
    s = np.array([x for x in single if not np.isnan(x)]); m = np.array(eq)
    cs, cm = boot(s, rng), boot(m, rng)
    print(f"{tag} L{L}: single {s.mean():+.2f} [{cs[0]:.2f}, {cs[1]:.2f}] n={len(s)} | equal-count {m.mean():+.2f} [{cm[0]:.2f}, {cm[1]:.2f}] n={len(m)}", flush=True)

if __name__ == "__main__":
    for tag, base, levels in [("orig", "results/regex-qwen3-32b/runs", [1, 2, 3, 4]),
                              ("prefill", "results/regex-qwen3-32b/prefill/runs", [1, 2, 3, 4]),
                              ("seed2026 prefill", "results/regex-qwen3-32b/seed2/runs", [4])]:
        for L in levels:
            run(tag, base, L)
