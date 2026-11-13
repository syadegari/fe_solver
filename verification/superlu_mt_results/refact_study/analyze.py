"""Summarise interleaved fresh/refact0 probe runs.

RESULT line fields: f1,f2 = total time of factorization 1,2 (perm+init+pdgstrf);
p,i,g = perm, init, pdgstrf of each.  Reported per thread count:
  f2        absolute cost of the second factorization (what a Newton step pays)
  f2/f1     within-process ratio; cancels process-to-process speed drift
  g2        pdgstrf only, isolating the saving that needs refact
Welch t compares refact0 against fresh.
"""
import re, sys, statistics as st
from collections import defaultdict

runs = defaultdict(list); failed = []
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith("RESULT"): continue
        if "FAILED" in line: failed.append(line.strip()); continue
        tok = line.split()
        rec = {"mode": tok[1]}
        for kv in tok[2:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                rec[k] = float(v)
        runs[(int(rec["nprocs"]), int(rec.get("fill", 0)), rec["mode"])].append(rec)

def ms(xs): return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else float("nan"))
def welch(a, b):
    if len(a) < 2 or len(b) < 2: return float("nan")
    va, vb = st.variance(a), st.variance(b)
    d = (va/len(a) + vb/len(b)) ** 0.5
    return (st.mean(a) - st.mean(b)) / d if d else float("inf")

keys = sorted({(k[0], k[1]) for k in runs})
for p, fill in keys:
    fr = runs.get((p, fill, "fresh"), []); rf = runs.get((p, fill, "refact0"), [])
    print(f"\n=== nprocs={p} fill={fill}  (n fresh={len(fr)}, n refact0={len(rf)}) ===")
    for name, rs in (("fresh", fr), ("refact0", rf)):
        if not rs: continue
        f2m, f2s = ms([r["f2"] for r in rs])
        qm, qs = ms([r["f2"]/r["f1"] for r in rs])
        g2m, g2s = ms([r["g2"] for r in rs])
        print(f"  {name:8s} f1={st.mean([r['f1'] for r in rs]):8.3f}  f2={f2m:8.3f}±{f2s:6.3f}  "
              f"f2/f1={qm:.4f}±{qs:.4f}  g2(pdgstrf)={g2m:8.3f}±{g2s:6.3f}  "
              f"max res2={max(r['res2'] for r in rs):.2e}")
    if fr and rf:
        if fr: print(f"  setup avoided by reuse: perm={st.mean([r['p2'] for r in fr]):.4f} s  "
                     f"init={st.mean([r['i2'] for r in fr]):.4f} s")
        for label, f in (("f2   ", lambda r: r["f2"]),
                         ("f2/f1", lambda r: r["f2"]/r["f1"]),
                         ("g2   ", lambda r: r["g2"])):
            a, b = [f(r) for r in fr], [f(r) for r in rf]
            ma = st.mean(a)
            print(f"  saving refact0 vs fresh on {label}: {100*(ma-st.mean(b))/ma:+6.2f}%  "
                  f"(Welch t={welch(a,b):+.2f})")
for f in failed: print("FAILED:", f)
