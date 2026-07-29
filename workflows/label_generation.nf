// §4.5 bulk label generation. Runs after the `qc` stage, which freezes the
// metabolite index every label row is hashed against (P13):
//
//   GENERATE_LABELS (per organism: active subspace -> Sobol design -> QP solves)
//
// No collect step -- the hive-partitioned parquet tree published under
// `${outdir}/labels` is the artifact, and pandas/pyarrow read it as one dataset.

include { GENERATE_LABELS } from '../modules/local/generate_labels/main'

workflow LABEL_GENERATION {
    take:
    ch_roster    // file: roster CSV (genome_id, model_path)
    ch_index     // file: frozen metabolite_index.json (from the qc stage)

    main:
    // Optional per-run inputs: a scales.json fallback for the §4.7 bands, and §4.6
    // focus weights from `cfs topup`. `[]` stages nothing and drops the flag.
    ch_scales = params.label_scales ? file(params.label_scales, checkIfExists: true) : []
    ch_focus  = params.label_focus_weights ? file(params.label_focus_weights, checkIfExists: true) : []

    // Same roster resolution as GROUNDTRUTH_QC: model_path is absolute or
    // relative to the roster file.
    ch_in = Channel.fromPath(ch_roster)
        .splitCsv(header: true)
        .map { row ->
            def model = file(row.model_path.startsWith('/') ? row.model_path : "${ch_roster.parent}/${row.model_path}", checkIfExists: true)
            [ [id: row.genome_id], model, ch_index, ch_scales, ch_focus ]
        }

    GENERATE_LABELS(ch_in)

    emit:
    shards   = GENERATE_LABELS.out.shards
    versions = GENERATE_LABELS.out.versions.first()
}
