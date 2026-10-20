# Handoff: 20HM v2 genomes — CI fix-up and the HPC labels/training chain

Written 2026-10-20 at a mid-task stop. Two open pieces of work:

- **A.** Make PR [timrozday-mgnify/genome-annotation#2](https://github.com/timrozday-mgnify/genome-annotation/pull/2)'s CI green. The user chose **full nf-core compliance**.
- **B.** Write the HPC scripts that turn the new CarveMe GEMs into the label roots and the four heads `workflows/pair-interactions` needs. The user copies the files to the cluster and runs them; the agent only writes the scripts.

## Background: what is already done

The user has better 20HM genomes at
`/Users/timrozday/Documents/mimicc/20hm-genome-collection/genomes/`. There are 24 members, keyed
by strain (`bacteroides_fragilis`, …). The manifest is `metadata/panel_genomes.tsv`:
21 are `core`, two are `alternative_strain` (`bacteroides_uniformis_strain2`/`_strain3`) and one
is `spike_in` (`salinibacter_ruber`). `methanobrevibacter_smithii` is archaeal.

**genome-annotation** (`/Users/timrozday/Documents/Pipelines/genome-annotation`, the user's repo):
- The branch `carveme-universe` is pushed and PR #2 is open. It adds an optional samplesheet
  column `universe`, which goes into `meta.universe` and then `carve -u`; it defaults to `bacteria`.
  The column is needed for the archaeon.
- **Uncommitted** on that branch, from the CI investigation:
  - `nextflow.config`: a `manifest` block (name/author/homePage/description/defaultBranch/
    nextflowVersion/version). Before this, nf-core lint crashed with `KeyError: manifest.name`.
  - `nextflow_schema.json`: lines 229-230. A misplaced `}` closed `$defs` early, so the file was
    **invalid JSON**. This came in with commit e53d160. The fix is verified (`json.load` lists all
    six `$defs`).
  - Keep both edits. Untracked `AGENTS.md`, `CLAUDE.md` and `.nf-test*` were there before; do
    not commit them.

**CarveMe run** (`~/Documents/surrogate-mgems_runs/20hm_v2_models/`, running in the background
on the laptop):
- Files:
  - `samplesheet.csv`: 24 rows, with `universe=archaea` for *M. smithii*.
  - `params.yml`: CarVeME only. It is a params file because a CLI `--run_x false` reaches the
    workflow as a truthy string.
  - `local.config`: Docker, 4 carves at once.
  - `roles.tsv`: strain_key → role.
- `run.sh` runs the pipeline. It then copies the models to
  `~/Documents/20hm_v2_carveme_models/<strain_key>.xml` and writes `roster_all.csv` (24) and
  `roster.csv` (the 21 core). It is resume-safe: re-run it if it died.
- `qc.sh` is **not yet run**. Run it after `run.sh`. It runs `cfs qc --no-memote` (the EGC gate)
  and `cfs freeze-index` on the core roster, writing `qc/metabolite_index.json`. The old 21-GEM
  index (`surrogate-mgems_runs/20hm/results/qc/metabolite_index.json`) **must not** be reused:
  the exchange set changes (P13).
- At the time of writing, 4 of 24 models were done (CarveMe 1.6.6, the same version as the old
  roster).
- To check progress: `tail ~/Documents/surrogate-mgems_runs/20hm_v2_models/run.log` and
  `ls .../results/*/carveme/*.xml | wc -l`.
- Before B goes to the cluster, also check that the archaeal carve succeeded and that `qc.sh`
  passes the EGC gate on 21/21.

## A. genome-annotation CI → full nf-core compliance

**Failing checks on PR #2.** All of them predate the PR; PR #1 failed the same ones.

| check | cause |
| --- | --- |
| `nf-core` (lint `--release`, because the base is `master`) | crashed first on `manifest.name`, then on the invalid schema (both fixed locally, uncommitted). It now reports **62 failures** |
| `pre-commit` | `end-of-file-fixer`/`trailing-whitespace`/prettier rewrite files, including **binary fixtures** under `tests/testdata/` (`.dmnd`, `.hmm`) and `nextflow.config`, `bin/*.py` |
| `template_version` | `template-version-comment.yml` needs `secrets.NF_CORE_BOT_AUTH_TOKEN`, which only the nf-core org has |
| `download` | `nf-core pipelines download` test; it failed on PR #1 too. Not yet diagnosed |
| `nf-test-changes` | was pending; check it |

**Lint with the pinned version.** CI installs `nf-core==` the `nf_core_version` in `.nf-core.yml`
(3.3.2). The local `nf-core` is 4.1.0 and gives different results. Use:

```bash
uvx --from nf-core==3.3.2 nf-core pipelines lint --release --markdown lint.md
```

**Trap:** nf-core caches `nextflow config` output in `~/.nextflow/nf-core/wf-config-cache-*.json`.
A stale entry made the `manifest.name` fix look as if it had not worked. Delete the cache entry
named in `nf-core -v ...` debug output when config edits seem ignored.

**The 62 failures (3.3.2, after the two local fixes), grouped:**
- `schema_params` (~27 unique): the schema documents template params the config does not define
  (`input`, `email`, `multiqc_*`, `genome`, `fasta`, `igenomes_*`, `custom_config_*`,
  `config_profile_*`, `version`, `hook_url`, `validate_params`, …). Going the other way, the
  config defines params the schema lacks: `samplesheet`, `singularity_cachedir`,
  `modules_testdata_base_path`, `prodigal_meta`, the `databases.*` tree and the chunksizes.
  Full compliance means restoring the standard template config params and adding the
  pipeline's own params to the schema.
  - Probably also rename `params.samplesheet` to nf-core's `params.input`. That ripples into
    `workflows/genomeannotation.nf`, `conf/test.config`, the tests and the docs, and the user's
    run scripts. Note it in the PR.
- `nextflow_config`:
  - `process.cpus/memory/time` not found: **`conf/base.config` is never included** by
    `nextflow.config`. That is a real bug, since the resource labels are not applied today.
    Including it changes default resources and errorStrategy, so check `conf/codon.config` and
    `conf/local.config` still win.
  - unpinned `nf-amazon`.
  - `validation.help.*` / `validation.summary.*`.
  - the custom-profile `includeConfig` lines (nf-core/configs).
  - `manifest.contributors`.
  - `homePage` not on github.com/nf-core: this is an nf-core naming rule, so an ignore in
    `.nf-core.yml` is legitimate even under full compliance.
  - `version` contains `dev` under `--release`.
  - `pipelines_testdata_base_path` default mismatch.
- `files_exist`: `CODE_OF_CONDUCT.md`, `assets/email_template.{html,txt}`,
  `assets/sendmail_template.txt`, `assets/nf-core-genomeannotation_logo_light.png`,
  `ro-crate-metadata.json`. `nf-core pipelines sync` / the template can regenerate them.
- `pipeline_todos`: ~39 template TODO strings in meta.yml files, `docs/*.md`, `conf/base.config`,
  `conf/test_full.config`, `awsfulltest.yml`, `tests/*/mock.nf` and local subworkflows.
- `version_consistency`: `1.0.0dev` is not numeric under `--release`.
- `nf_test_content`: `tests/default.nf.test` does not snapshot `versions.yml`. Fixing it means
  updating snapshots (`nf-test test tests/default.nf.test --profile +docker --update-snapshot`,
  ~10 min, needs the mini DBs from `tests/scripts/make_mini_databases.sh`).
- `readme`: missing badges.
- Also: `rocrate_readme_sync`, `included_configs`, `files_unchanged`, `multiqc_config`.
- Module/subworkflow lint:
  - `fastqc`, `kofamscan`, `multiqc`, `pyrodigal`: "new version available" and
    "local copy modified". CLAUDE.md says fastqc/multiqc are unused, so **delete them**
    (`nf-core modules remove`). kofamscan and pyrodigal were locally modified (`hmmsearch` too,
    per commit f13d4b5), so either upstream the change, use `nf-core modules patch` to record it,
    or update and re-apply.
  - Local subworkflows `fetchdb` / `utils_nfcore_genomeannotation_pipeline`: TODOs, a missing
    `meta.yml`, include issues.
  - nf-core subworkflows `utils_*`: new versions available (`nf-core subworkflows update`).

**Suggested order:**
1. Commit the two existing fixes.
2. pre-commit: add `tests/testdata/` to the hooks' `exclude`, so binary fixtures and md5
   snapshots are never rewritten, then `pre-commit run --all-files` and commit the result.
3. Delete `template-version-comment.yml`, which can never pass outside nf-core. Also drop the
   AWS workflows, which need Tower secrets; CLAUDE.md lists them as removable.
4. Schema/config drift.
5. Modules and subworkflows.
6. TODOs, files and badges.
7. Re-lint with 3.3.2 until clean.
8. Diagnose `download`.

**Which PR.** The user said full compliance is a large diff and belongs in a separate PR. Branch
from `carveme-universe` (or from master after #2 merges). Keep #2 for the `universe` feature plus
the two crash fixes. Ask the user if unsure.

Run the e2e nf-test before claiming green. The user's live CarveMe run executes from this
checkout's working tree, so do not switch branches in
`/Users/timrozday/Documents/Pipelines/genome-annotation` while it is running. Use a git worktree.

## B. HPC labels + training chain for the new genomes

**What it must produce.** The pair-interactions pipeline needs these, per arm, as in
`workflows/pair-interactions/nextflow.config` `params.arms` and `stage_data.sh`:

| arm | label root | Head A | Head B |
| --- | --- | --- | --- |
| `fba` | `labels_p4` | `value_p4r2` | `behaviour_p4r2` |
| `ceq1mM` | `labels_i3` (`--inhibition ceq_1.0.json` = `{"default": 1.0}`) | `value_i3_cap` | `behaviour_i3` |

Reuse those exact names so `stage_data.sh` and the pipeline work unchanged. Every input is the
new `qc/metabolite_index.json` and `roster.csv` from `20hm_v2_models/`.

**The recipe**, reconstructed from `~/Documents/surrogate-mgems_runs/20hm_bands/`. Read
`then_i3.sh`, `r2.sh`, `cap_check.sh`, `one_organism_*.sh`, `make_a1.sh`, `n15.sh`,
`make_label_pool.py` and `one_organism_media.sh`.

1. **Base labels**, one task per organism:
   ```
   cfs generate --roster <one-row roster> --index <idx> --outdir <root> --n-media 4000
   ```
   For `labels_i3`, add `--inhibition ceq_1.0.json`.
   - Pass **`--mid-mu 0`**. labels_p4 predates the B2 mid-`mu` stratum, whose default is now
     0.15; CLAUDE.md says to set 0 for a new label root.
   - **Omit `--scales`**. The old `scales.json` is keyed by old genome ids; the §4.7 demand probe
     (on by default) anchors new GEMs itself.
   - Check 3 eps shards per organism (`part.parquet` under `genome_id=*/eps=*`).
2. **Round 1 community pass**, per organism: the same command plus
   `--n-media 800 --bg-perturb 0.9 --round 1 --seed 1`, which writes `part.round1.parquet`.
3. **Level-1 trial media:**
   ```
   cfs community-holdout make --roster roster.csv --labels <labels_p4> --index <idx> \
       --communities "<communities.txt>" --n-media 200 --seed 7000 --out holdout_community
   ```
   - The old `communities.txt` uses old ids. Write a new one with the same shape: semicolon-
     separated member lists, 10 communities of sizes 2,2,2,2,2,3,3,5,10,21 drawn from the 21 core
     with a fixed seed.
   - Also write `communities15.txt`: 16 distinct 15-member subsets, used for round 2.
   - Write both as committed inputs and record the seed.
4. **Heads per arm.** `train-value` exits 1 when the M3 gate is unmet; that is not a failure.
   Gate on `diagnostics.json` existing.
   - Head A, frozen level-1 tangents + repair:
     ```
     cfs train-value --labels <root> --index <idx> --out <value_dir> --arch groupmax-u \
         --width 1 --depth 1 --epochs 0 --lr 3e-3 --w-grad 10 --gm-group 1000 \
         --gm-temp 0.01 --gm-eval-temp 0.0001 --gm-init labels --gm-select level1 \
         --gm-trial-media holdout_community/community_holdout.npz --gm-repair --seed 0
     ```
     `_cap` is just this, trained after M17's `W_CAP=1e5`, which is now the code default.
   - Head B:
     ```
     cfs train-behaviour --labels <root> --index <idx> --out <behaviour_dir> \
         --width 256 --depth 3 --epochs 600 --lr 1e-3 --seed 0
     ```
5. **The fba arm's round 2 (the `r2` in `value_p4r2`):**
   - Train interim `value_p4` + `behaviour_p4` on base + round 1.
   - Run `cfs community --roster roster.csv --labels labels_p4 --value value_p4 --behaviour
     behaviour_p4 --communities "$(cat communities15.txt)" --seed {0,100} --out
     community_n15_s{seed}`. These are true-LP trajectories, the expensive part.
   - Build the pool: `make_label_pool.py label_pool_n15.npz 5 community_n15_s0
     community_n15_s100`. It reads `holdout_community/community_holdout.npz` relative to cwd.
     Copy it into the HPC bundle; it is ~20 lines.
   - Label the pool per organism: `cfs generate --media label_pool_n15.npz --round 2`.
   - Retrain both heads as `value_p4r2` / `behaviour_p4r2`. P14: `x_scale` moves with a round, so
     both heads must be rebuilt together.
   - The i3 arm has **no** round 2 (`labels_i3` = base + round 1).
6. Optional checks, laptop-sized:
   - `cfs community-holdout score --holdout ... --value <dir>`.
   - `cfs community` over `communities.txt` at seeds 0/100/200 (the M5 benchmark).

**Shape of the HPC deliverable.** It must be Slurm-friendly and match the house style:
`workflows/pair-interactions/{run_hpc.sh,setup.sh,hpc.example.config}`. The venv comes from
`setup.sh` (`uv venv`, `pip install -e ".[data,jax]"`).

Two candidate routes:
- **Extend the root pipeline.** `main.nf --stage labels` (`workflows/label_generation.nf`)
  already fans `cfs generate` out per organism, and `--stage sweep` trains heads. But the labels
  stage runs in the `surrogate-mgem-data` container, the round/inhibition/media flags are only
  partly wired, and there is no stage for round 2 or the community runs.
- **A new `workflows/hpc-labels/` Nextflow pipeline on the host venv**, like pair-interactions.
  This is probably simpler. The DAG would be:
  ```
  GENERATE(base)
    -> GENERATE(round1)
    -> HOLDOUT
    -> TRAIN(arm)
    -> [fba: COMMUNITY_N15 x2 -> POOL -> GENERATE(round2) -> TRAIN(p4r2)]
  ```
  with per-organism fan-out for every GENERATE, and `OMP_NUM_THREADS=1` / one XLA thread per
  task (CLAUDE.md: never let cobra or the solvers parallelise inside a task).

Pick one, confirm with the user if it is not obvious, and include a `test` profile that runs 2
organisms at tiny `--n-media` to check the plumbing.

**Cost guidance** (laptop; base and round 2 are measured, the rest are estimates):
- Base labels: ~1 h/organism. It dies mid-shard on a full disk; delete a partial shard, do not
  resume it.
- Round 1: ~20% of base (800 vs 4000 media), estimated.
- Round 2: ~8.5 min/organism.
- `train-value` with `--epochs 0`: minutes.
- `train-behaviour`, 600 epochs: not recorded; budget an hour.
- `cfs community` on 16 × 15-member communities: the dominant cost, hours.

`labels_*` take ~30 MB/organism/root. Checkpoints take ~100 MB each.

**After the cluster run:**
- Bundle with
  `GEM_DIR=~/Documents/20hm_v2_carveme_models bash workflows/pair-interactions/stage_data.sh
  <run_root> <bundle> <members...>`.
- Update the strain-key members in `workflows/pair-interactions/nextflow.config`. The defaults
  are old accession ids (`CP040530.1,CP070062.1`).
- Old → new mapping, from the manifest's `accession` column:
  - `CP040530.1` → `bacteroides_thetaiotaomicron`
  - `CP070062.1` → `coprococcus_comes`
  - `CR626927.1` → `bacteroides_fragilis`
- Six old ids do not appear in the new manifest's accessions: `AAXE02` (Dorea), `ABCC02`,
  `ABFX02`, `ABYJ02`, `DACTBY01`, `FNPN01`. Do not assume a mapping; read the manifest.

## Rules from the project worth repeating
- One `index_hash` across every shard of a root (P13). Rebuild both heads whenever a round is
  added (P14).
- Re-measure whether to calibrate on every new label root. `--gm-repair` already forces identity.
- Score a label round against a matched retrain, not against the checkpoint it started from.
- A feature written in the design spec is not evidence it runs. Smoke-test every CLI invocation
  in the scripts with the `test` profile before handing them over.
