"""Community-regime media to LABEL, from trajectories the M5 benchmark never scores.

§8.6d: Head B's residual is coverage of the states §8.1 visits (NN distance in `x`
predicts `dc_rel` at Spearman +0.673). The *selection* pool (`--gm-trial-media`) may
reuse the benchmark communities, because no cut takes a label there; labelling costs
solves and would train on the benchmark. The n=15 runs are a disjoint community set
on their own medium draws, so their forward-pass states are the same regime and none
of them is scored.

    python make_label_pool.py --holdout community_holdout.npz --stride 5 \
        --out label_pool_n15.npz <community run dir> ...

Taken from 20hm_bands/make_label_pool.py; the holdout is a flag here rather than a
path relative to the cwd.
"""
import argparse, glob
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+", help="`cfs community` output dirs")
p.add_argument("--holdout", required=True, help="community_holdout.npz: the exchange order + index_hash")
p.add_argument("--stride", type=int, default=5, help="keep every Nth trajectory state")
p.add_argument("--out", required=True)
a = p.parse_args()

ref = np.load(a.holdout, allow_pickle=True)
ex = [str(e) for e in ref["exchanges"]]

media = []
for run in a.runs:
    for f in sorted(glob.glob(f"{run}/trajectory_*.npz")):
        d = np.load(f, allow_pickle=True)
        assert [str(e) for e in d["exchanges"]] == ex, f
        media.append(d["c_true"][:: a.stride])
if not media:
    raise SystemExit(f"no trajectory_*.npz under {a.runs}")
media = np.concatenate(media)
np.savez_compressed(a.out, media=media, exchanges=np.array(ex), index_hash=ref["index_hash"])
print(a.out, media.shape)
