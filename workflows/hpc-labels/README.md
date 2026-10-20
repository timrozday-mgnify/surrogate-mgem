# hpc-labels — label roots and heads for a new GEM roster

Turns a directory of CarveMe GEMs into the two label roots and the four checkpoints
`workflows/pair-interactions` consumes. It exists because that pipeline's inputs
(`labels_p4`, `value_p4r2`, `behaviour_p4r2`, `labels_i3`, `value_i3_cap`,
`behaviour_i3`) were produced by a pile of laptop shell scripts under
`20hm_bands/`; this is the same recipe, fanned out per organism and resumable.

The output names are the ones `pair-interactions/stage_data.sh` looks for. Changing
them means changing that script too.

## What it builds

| arm | ground truth | label root | Head A | Head B |
| --- | --- | --- | --- | --- |
| `fba` | plain FBA | `labels_p4` (base + round 1 + round 2) | `value_p4r2` | `behaviour_p4r2` |
| `ceq1mM` | §13.11 product inhibition, `c^eq` = 1 mM | `labels_i3` (base + round 1) | `value_i3_cap` | `behaviour_i3` |

plus `communities.txt` and `communities15.txt` (the member lists everything
downstream is quoted against), `holdout_community/` (the level-1 trial media) and
`label_pool_n15.npz` (round 2's media).

A `c^eq` makes a **different label root**, not a flag: Head A gains a per-metabolite
`theta` channel, so the two arms share nothing but the index. `c^eq` = 1 mM and not
0.1 mM because 0.1 starved 13 of 21 organisms on the old roster — median `mu_max`
0.000 — which emptied the Sobolev term and was the whole cosine tail.

## Run

```bash
bash workflows/pair-interactions/setup.sh                     # the shared repo venv
bash workflows/hpc-labels/stage_inputs.sh \
    ~/Documents/20hm_v2_carveme_models \
    ~/Documents/surrogate-mgems_runs/20hm_v2_models/qc/metabolite_index.json \
    ~/Documents/surrogate-mgems_runs/20hm_v2_models/roster.csv \
    labels-data
rsync -a labels-data/ hpc:<shared storage>/labels-data/

cp workflows/hpc-labels/hpc.example.config workflows/hpc-labels/hpc.config  # edit it
sbatch workflows/hpc-labels/run_hpc.sh
```

Smoke test first — minutes, two organisms, every CLI invocation for real:

```bash
PROFILE=slurm,test sbatch workflows/hpc-labels/run_hpc.sh \
    --organisms <id>,<id> --outdir <scratch>/labels-test
```

`--outdir` on the command line, always, for the test profile: the repo-root
`nextflow.config` is the launch-directory config and overrides what a profile sets.
The same applies to `process.resourceLimits`, which `conf/base.config` pins to
`--max_memory`; the test profile therefore sets the training tasks' memory outright.

The index **must** be the one `cfs freeze-index` froze on this roster. A new GEM set
has a different exchange set, so an old `metabolite_index.json` is a different model
and every shard carries its hash (P13).

## The DAG

```
GENERATE(arm, organism)        base 4000 media + the round-1 community pass
COMMUNITIES                    both member lists, drawn once off the roster (seeded)
HOLDOUT                        level-1 trial media; one set serves both arms
TRAIN_{VALUE,BEHAVIOUR}_INTERIM          the throwaway fba heads
  COMMUNITY(community, seed)   true-LP n=15 trajectories   <- the dominant cost
  POOL -> GENERATE_MEDIA(organism)       round 2 (§8.6d)
ASSEMBLE(arm)                  the published root
TRAIN_{VALUE,BEHAVIOUR}        the four shipped heads
```

Base and round 1 are one task per organism rather than two, because `--round 1`
rewrites that organism's subspace sidecars and the two passes must not race for
them. Round 2 is `cfs generate --media`, which labels an explicit medium list and
writes no sidecars, so only the base pass contributes them — which is what lets
ASSEMBLE merge the two without a collision.

`cfs community` is fanned out one task per (community, seed): it names its
trajectory files after the members, so nothing needs disambiguating, and the 32
tasks are the only thing in the run worth parallelising hard.

### Why the round

Round 1 (`--bg-perturb 0.9`) is the community regime: §4.3 holds the background at
one rich level while a community medium is a draw over the **union** of the members'
active subspaces, so a member sees ~30 of its background metabolites in bands at
once. Round 2 labels the states the composition actually visits, taken from n=15
communities that are deliberately **not** the benchmark — labelling a benchmark
medium would train on the thing being scored. It halves the whole-path
nearest-neighbour distance and the composition's overall error; it does not move the
far cells.

Both heads are rebuilt from the round-2 root together. `x_scale` is `_kink_scale`
over the *training* rows, so a round moves the input coordinate (P14) and a reused
head and a new root disagree about it — `Surrogate.__init__` checks, which is what
the check is for.

### Head A

Frozen label tangents, not a trained head: `--epochs 0 --gm-init labels
--gm-select level1 --gm-repair --gm-eval-temp 1e-4`. The selected tangents beat
every trained head at every community size and lose nowhere; `--gm-eval-temp` ships
the head colder than it selected cuts at, which removes the repair's uniform
smoothing lift — the arithmetic error a whole branch of loss terms was
compensating for. Do not reach for `--w-rel`, `--w-under`, `--w-tau`, the mid-`mu`
stratum or the output calibration here.

`cfs train-value` **exits 1 when the M3 gate is unmet**. That is an outcome recorded
in `diagnostics.json`, not a failure, so the task tolerates it and gates on the file.

## Cost

Per organism per arm: base ~1 h, round 1 ~20% of that, round 2 ~8.5 min — measured
on a laptop core at 4000 media. `cfs train-value` at `--epochs 0` is minutes but
loads every organism's shards at once (hence 48 GB); `train-behaviour` at 600 epochs
is the longer of the two. The 32 `cfs community` tasks are the wall clock.

Disk: ~30 MB an organism a root, and the work directory holds another copy, so
budget ~10 GB for 21 organisms over two arms plus ~0.3 GB a checkpoint. `cfs
generate` dies mid-shard on a full disk; a failed task's work directory is thrown
away, so there is no partial shard to clean up by hand.

## After the run

```bash
GEM_DIR=~/Documents/20hm_v2_carveme_models \
  bash workflows/pair-interactions/stage_data.sh <outdir> <bundle> <members...>
```

and set `params.members` in `workflows/pair-interactions/nextflow.config` to the new
genome ids — the defaults are the old roster's accessions.

Optional, laptop-sized, not in the pipeline because neither is an input to anything:

```bash
cfs community-holdout score --holdout holdout_community/community_holdout.npz --value value_p4r2 --out value_p4r2
cfs community --roster roster.csv --labels labels_p4 --value value_p4r2 \
    --behaviour behaviour_p4r2 --communities "$(cat communities.txt)" --seed 0 --out community_s0
```

The second is the M5 benchmark. Quote it over paired medium draws (0/100/200) and
never as the median of three: a single draw has a 6x sampling error on a small
community, and one draw of the full roster is a 448x outlier on one cell.
