// M0 + M1 ground-truth QC (plan §2, §3.0, §5.2 — gates V0/V1). Runs before any
// label generation or training:
//
//   QC_MODELS (EGC + MEMOTE, freeze metabolite index)
//     -> DEGENERACY_SURVEY (per organism, exchange-FVA)
//        -> COLLECT_D4 (roster-wide D4 recommendation)
//
// The D4 recommendation is advisory; the human records the decision (plan §5).

include { QC_MODELS        } from '../modules/local/qc_models/main'
include { DEGENERACY_SURVEY } from '../modules/local/degeneracy_survey/main'
include { COLLECT_D4       } from '../modules/local/collect_d4/main'

workflow GROUNDTRUTH_QC {
    take:
    ch_roster    // file: roster CSV (genome_id, model_path)

    main:
    ch_versions = Channel.empty()

    // Stage the GEMs flat and rewrite the roster to bare filenames, exactly as the
    // training workflow does (read_roster resolves model_path against the roster).
    ch_rows = Channel.fromPath(ch_roster).splitCsv(header: true)
    ch_models = ch_rows
        .map { row -> file(row.model_path.startsWith('/') ? row.model_path : "${ch_roster.parent}/${row.model_path}", checkIfExists: true) }
        .collect()
        .map { models -> [ models ] }   // wrap: combine() would otherwise spread the list
    ch_flat_roster = ch_rows
        .map { row -> "${row.genome_id},${file(row.model_path).name}" }
        .collectFile(name: 'roster.csv', seed: 'genome_id,model_path', newLine: true, sort: true)
        .first()

    // M0: QC the whole roster in one task and freeze the metabolite index.
    ch_qc_in = Channel.of([ [id: 'roster'] ])
        .combine(ch_flat_roster)
        .combine(ch_models)
    QC_MODELS(ch_qc_in)
    ch_versions = ch_versions.mix(QC_MODELS.out.versions)

    // M1: fan out the degeneracy survey one task per genome (FVA is the cost).
    ch_survey_in = ch_rows.map { row ->
        def model = file(row.model_path.startsWith('/') ? row.model_path : "${ch_roster.parent}/${row.model_path}", checkIfExists: true)
        [ [id: row.genome_id], model ]
    }
    DEGENERACY_SURVEY(ch_survey_in)
    ch_versions = ch_versions.mix(DEGENERACY_SURVEY.out.versions.first())

    // Collect per-organism surveys into one roster-wide D4 recommendation.
    ch_collect_in = DEGENERACY_SURVEY.out.survey
        .map { meta, csv -> csv }
        .collect()
        .map { csvs -> [ [id: 'roster'], csvs ] }
    COLLECT_D4(ch_collect_in)
    ch_versions = ch_versions.mix(COLLECT_D4.out.versions)

    emit:
    index          = QC_MODELS.out.index
    recommendation = COLLECT_D4.out.recommendation
    versions       = ch_versions
}
