// M3b/§7.4: the Head A architecture + scale sweep.
//
//   TRAIN_VALUE / BASELINE_RF (one task per samplesheet row x organism)
//     -> COLLECT_VALUE_METRICS (one sweep_leaderboard.csv, one row per cell)
//
// Cells come from a samplesheet (`--sweep`), not from `Channel.fromList` over a
// param per axis as in SURROGATE_TRAINING: the axes here are not a clean cross
// product (`--phi-hidden`/`--k-code` are deepset-only, `--n-estimators`/`--delta`
// are rf-only, and the "5x rows" arm is a different label root), and a row's flag
// string being literal keeps each cell's exact command greppable.
// `examples/value_sweep/make_sweep.py` generates the samplesheet.

include { TRAIN_VALUE           } from '../modules/local/train_value/main'
include { BASELINE_RF           } from '../modules/local/baseline_rf/main'
include { COLLECT_VALUE_METRICS } from '../modules/local/collect_value_metrics/main'

workflow VALUE_SWEEP {
    take:
    ch_sweep     // file: sweep samplesheet CSV (cell_id, arch, labels, args)
    ch_index     // file: frozen metabolite_index.json (from the qc stage)

    main:
    // `labels` is absolute or relative to the samplesheet, matching the roster's
    // model_path convention elsewhere.
    // One task per (cell, organism), not per cell. The organism axis is a vmap
    // axis, not a modelling choice: only the shared-trunk `deepset` pools anything
    // across it, so every other arch trains the same heads whether they are stacked
    // 21-wide or 1-wide -- and 21 short jobs beat one long one on a queue. The
    // organisms come from the label root's own per-genome shard dirs (the sheet's
    // `labels` column differs per row); a root with none -- the stub -- falls back
    // to a single whole-stack task, which is also the escape hatch if a cell ever
    // wants the old behaviour.
    ch_cells = Channel.fromPath(ch_sweep)
        .splitCsv(header: true)
        .flatMap { row ->
            def labels = file(row.labels.startsWith('/') ? row.labels : "${ch_sweep.parent}/${row.labels}", checkIfExists: true)
            def gids = row.arch == 'deepset'
                ? []
                : labels.listFiles().findAll { it.isDirectory() }.collect { it.name }.sort()
            def meta = [arch: row.arch, cell_args: row.args ?: '']
            (gids ?: [null]).collect { gid ->
                [ meta + [id: gid ? "${row.cell_id}__${gid}" : row.cell_id, cell: row.cell_id, organism: gid], labels, ch_index ]
            }
        }
        .branch { meta, labels, index ->
            rf: meta.arch == 'rf'
            nn: true
        }

    TRAIN_VALUE(ch_cells.nn)
    BASELINE_RF(ch_cells.rf)

    ch_results = TRAIN_VALUE.out.results.mix(BASELINE_RF.out.results)

    COLLECT_VALUE_METRICS(
        ch_results
            .map { meta, d -> d }
            .collect()
            .map { dirs -> [ [id: 'sweep_leaderboard'], dirs ] }
    )

    emit:
    results     = ch_results
    leaderboard = COLLECT_VALUE_METRICS.out.leaderboard
    versions    = TRAIN_VALUE.out.versions
                      .mix(BASELINE_RF.out.versions, COLLECT_VALUE_METRICS.out.versions)
                      .first()
}
