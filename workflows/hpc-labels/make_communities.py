"""The two community lists every downstream step is quoted against, drawn once.

`communities.txt` is the M5 benchmark shape: sizes 2,2,2,2,2,3,3,5,10 and the whole
roster. `communities15.txt` is 16 distinct 15-member subsets -- the round-2 label pool
comes from *their* trajectories, deliberately disjoint from the benchmark so no label
is taken at a medium the benchmark then scores (§8.6d).

Seeded, so a list is a property of (roster, seed) and not of when it ran.

    python make_communities.py roster.csv 20251020 communities.txt communities15.txt
"""
import argparse, csv, random

p = argparse.ArgumentParser()
p.add_argument("roster")
p.add_argument("seed", type=int)
p.add_argument("out_bench")
p.add_argument("out_n15")
p.add_argument("--n15-sets", type=int, default=16)
p.add_argument("--n15-size", type=int, default=15)
a = p.parse_args()

with open(a.roster) as fh:
    gids = sorted(r["genome_id"] for r in csv.DictReader(fh))
rng = random.Random(a.seed)


def draw(sizes, want):
    """`want` distinct member sets of the given sizes; duplicates only if the roster
    is too small to avoid them (which is the `test` profile, not a real run)."""
    seen, out = set(), []
    for n in sizes:
        n = min(n, len(gids))
        for _ in range(1000):
            c = tuple(sorted(rng.sample(gids, n)))
            if c not in seen:
                break
        seen.add(c)
        out.append(c)
    assert len(out) == want
    return out


bench = draw([2, 2, 2, 2, 2, 3, 3, 5, 10, len(gids)], 10)
n15 = draw([a.n15_size] * a.n15_sets, a.n15_sets)

for path, cs in ((a.out_bench, bench), (a.out_n15, n15)):
    with open(path, "w") as fh:
        fh.write(";".join(",".join(c) for c in cs))
    print(path, len(cs), "communities", [len(c) for c in cs])
