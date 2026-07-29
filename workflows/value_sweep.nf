// M3b/§7.4: the Head A architecture + scale sweep.
//
//   TRAIN_VALUE / BASELINE_RF (one task per samplesheet row)
//     -> COLLECT_VALUE_METRICS (one sweep_leaderboard.csv)
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
    ch_cells = Channel.fromPath(ch_sweep)
        .splitCsv(header: true)
        .map { row ->
            def labels = file(row.labels.startsWith('/') ? row.labels : "${ch_sweep.parent}/${row.labels}", checkIfExists: true)
            [ [id: row.cell_id, arch: row.arch, cell_args: row.args ?: ''], labels, ch_index ]
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
