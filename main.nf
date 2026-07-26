// surrogate-mgem training pipeline entry point. Thin: validate params, resolve the
// roster, call the workflow. See README.md for the model and CLAUDE.md for layout.

include { SURROGATE_TRAINING } from './workflows/surrogate_training'
include { GROUNDTRUTH_QC     } from './workflows/groundtruth_qc'

workflow {
    // --- param validation (imperative, like the reference pipeline) -----------
    if (params.help) {
        log.info """
        surrogate-mgem

        Required:
          --roster <file>        Roster CSV with genome_id, model_path columns.

        Stages (--stage):
          qc     M0+M1 ground-truth QC: EGC + MEMOTE, freeze metabolite index,
                 exchange-FVA degeneracy survey (decides D4). Run this first.
          train  the training sweep (default).

        Key params (see nextflow.config for all + defaults):
          --outdir, --num_shards, --n_communities, --n_communities_augment,
          --active_rounds, --hidden_configs, --n_models_list, --train_sizes
        """.stripIndent()
        return
    }
    if (!params.roster) {
        error "Provide --roster <roster.csv> (columns: genome_id, model_path)."
    }
    if (!(params.stage in ['qc', 'train'])) {
        error "Unknown --stage '${params.stage}' (expected 'qc' or 'train')."
    }

    ch_roster = file(params.roster, checkIfExists: true)

    if (params.stage == 'qc') {
        GROUNDTRUTH_QC(ch_roster)
        ch_versions = GROUNDTRUTH_QC.out.versions
    } else {
        if ((params.num_shards as int) < 1) {
            error "num_shards must be >= 1 (got ${params.num_shards})."
        }
        if ((params.active_rounds as int) < 0) {
            error "active_rounds must be >= 0 (got ${params.active_rounds})."
        }
        SURROGATE_TRAINING(ch_roster)
        ch_versions = SURROGATE_TRAINING.out.versions
    }

    ch_versions
        .unique()
        .collectFile(name: 'software_versions.yml', storeDir: "${params.outdir}/pipeline_info")
}
