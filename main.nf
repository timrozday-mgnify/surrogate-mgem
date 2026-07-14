// surrogate-mgem training pipeline entry point. Thin: validate params, resolve the
// roster, call the workflow. See README.md for the model and CLAUDE.md for layout.

include { SURROGATE_TRAINING } from './workflows/surrogate_training'

workflow {
    // --- param validation (imperative, like the reference pipeline) -----------
    if (params.help) {
        log.info """
        surrogate-mgem training sweep

        Required:
          --roster <file>        Roster CSV with genome_id, model_path columns.

        Key params (see nextflow.config for all + defaults):
          --outdir, --num_shards, --n_communities, --n_communities_augment,
          --active_rounds, --hidden_configs, --n_models_list, --train_sizes
        """.stripIndent()
        return
    }
    if (!params.roster) {
        error "Provide --roster <roster.csv> (columns: genome_id, model_path)."
    }
    if (params.num_shards < 1) {
        error "num_shards must be >= 1 (got ${params.num_shards})."
    }
    if (params.active_rounds < 0) {
        error "active_rounds must be >= 0 (got ${params.active_rounds})."
    }

    ch_roster = file(params.roster, checkIfExists: true)
    SURROGATE_TRAINING(ch_roster)

    SURROGATE_TRAINING.out.versions
        .unique()
        .collectFile(name: 'software_versions.yml', storeDir: "${params.outdir}/pipeline_info")
}
