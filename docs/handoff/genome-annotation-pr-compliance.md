# PR body draft: `nf-core-compliance` → `carveme-universe`

Worktree: `~/Documents/Pipelines/genome-annotation-compliance`, branch
`nf-core-compliance`, 8 commits, `42 files changed, 311 insertions(+), 2424 deletions(-)`.
Not pushed — say the word.

Base it on `carveme-universe` (stacked on [PR #2](https://github.com/timrozday-mgnify/genome-annotation/pull/2)) so the
diff is only the compliance work. If #2 merges first, rebase onto `master` and retarget.

---

## Make CI green

`nf-core pipelines lint` went **62 failures → 0** (209 passed, 49 warnings, 9
ignored), and three CI jobs that could never pass were removed. Measured with
the pinned `nf-core==3.3.2` from `.nf-core.yml`, not the newer local install —
they disagree.

The two crash fixes that made linting possible at all (the `manifest` block and
the invalid `nextflow_schema.json`) are in #2, not here.

| was failing | now |
| --- | --- |
| `nf-core` lint, 62 | 0 failures |
| `pre-commit` | passes; fixtures excluded, one real YAML bug fixed |
| `template_version` | job deleted (needs an nf-core org secret) |
| `download` | job deleted (resolves the pipeline by its nf-core registry name) |
| `nf-test-changes` | unaffected |

## Read this before merging

**`params.samplesheet` is now `params.input`.** It is the nf-core standard name
and the repo was already inconsistent — the deleted samplesheet parser read
`params.input`, which no config defined. Anything that passes the samplesheet
must be updated:

```diff
-samplesheet: samplesheet.csv
+input: samplesheet.csv
```

That includes `~/Documents/surrogate-mgems_runs/20hm_v2_models/params.yml`. The
20HM v2 CarveMe run already completed against the old name, so nothing in
flight breaks; the next run after merging needs the one-line change.

## Two latent bugs this surfaced

**`conf/base.config` was never included by `nextflow.config`.** Every
`process_single/low/medium/high` label on every module did nothing — processes
ran on Nextflow's own defaults with `errorStrategy` terminate rather than the
retry policy. Now included, before `profiles` so `local`, `codon` and any `-c`
site config still override it. This changes default resources, which is a
behaviour change worth a second pair of eyes.

**`nextflow_schema.json` validates nothing.** `validateParameters()` is only
called from `UTILS_NFSCHEMA_PLUGIN`, and nothing invokes it — `main.nf` calls
`GENOMEANNOTATION()` directly, with no `PIPELINE_INITIALISATION`. That is how
the schema could sit as invalid JSON through two commits. This PR makes the
schema correct and sets `validation.help` (which nf-schema reads from config
directly, so `--help` does work), but **does not** wire validation back up:
that changes runtime behaviour and deserves its own PR.

## What was deleted, and why it was safe

All dead by inspection, and all four on CLAUDE.md's own "nf-core Scaffolding to
Remove" list:

- `subworkflows/local/utils_nfcore_genomeannotation_pipeline/` — nothing
  includes it. Its samplesheet parser is the template's fastq one
  (`fastq_1`/`fastq_2`/`single_end`) for a pipeline whose samplesheet is
  `sample,fastx`, and it was the only reader of `params.input`/`params.genome`.
- `modules/nf-core/fastqc`, `modules/nf-core/multiqc` — via
  `nf-core modules remove`, so `modules.json` stays consistent.
- `conf/igenomes.config`, `conf/igenomes_ignored.config` — never included.
- `.github/workflows/{awstest,awsfulltest,download_pipeline,template-version-comment}.yml`.

`branch.yml` stays: it is gated on `github.repository == 'nf-core/...'`, so it
no-ops rather than failing.

## The 9 lint ignores

Each is a rule about pipelines *published in the nf-core organisation*, listed
individually in `.nf-core.yml` so a new failure in the same test still surfaces:
`manifest.homePage` must start with `github.com/nf-core/`; `custom_config_*`
loads nf-core/configs (this pipeline ships `codon`/`local` instead);
`CODE_OF_CONDUCT.md` governs nf-core's own community; and the three email
templates plus the nf-core logo are read only by `completionEmail`, whose caller
is gone.

**`linting.yml` no longer runs `--release`.** nf-core develop on `dev` and
release from `master`, so their template reads "base branch is master" as "this
is a release PR". Here `master` *is* the development branch, so every PR was
linted as a release and failed on `manifest.version` containing `dev` and on
`version_consistency` wanting a numeric version — complaints about a release
that has not happened.

## Also

- The four locally-modified nf-core modules are recorded with
  `nf-core modules patch`, so lint passes and a future
  `nf-core modules update` replays the patch instead of silently reverting it.
- The pipeline now writes `pipeline_info/genomeannotation_software_versions.yml`.
  `softwareVersionsToYAML` was imported and never called, so no run recorded
  what it ran with. The e2e test snapshots it.
- `*_chunksize` are integer bytes (`2097152`) rather than `2. MB` — which is
  what CLAUDE.md already documents them as, and what the schema can express.
- `conf/local.config` set `params.hmmsearch_chunksize`, which nothing reads.

## The one red check that remains

The e2e nf-test will fail until `tests/default.nf.test.snap` is regenerated,
because the pipeline now writes a software-versions file. Regenerating it needs
the `gapseq:2.0.1` image, which is a bare local tag built from
`docker/gapseq/Dockerfile` and cannot be pulled -- so neither this machine nor
GitHub Actions can run that test today. Pre-existing: the committed snapshot
has 39 gapseq entries, so it was recorded somewhere that had the image.

On a machine with it:

```bash
docker build -t gapseq:2.0.1 docker/gapseq
nf-test test tests/default.nf.test --profile +docker --update-snapshot
git diff --stat tests/default.nf.test.snap   # sanity-check before committing
```

That last line matters: `--update-snapshot` re-recorded all four snapshots from
a *failed* run here, cutting the file from 251 lines to 20.

## Left undone

- **29 `pipeline_todos` warnings**: `TODO nf-core: Add a description...` in the
  local modules' `meta.yml`/`environment.yml`, `docs/{usage,output}.md`,
  `conf/{base,test_full}.config` and `tests/nextflow.config`. Real
  documentation debt, no CI effect, and writing those descriptions is a job for
  whoever knows what the modules do.
- Wiring parameter validation back up (see above).
- Upstreaming the four module patches to nf-core/modules.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
