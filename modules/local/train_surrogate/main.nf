// Train one sweep cell: a static ensemble for one community at a given architecture
// (meta.hidden), ensemble size (meta.n_models) and training-data cap (meta.n_train).
// Writes train_metrics.json + held-out predictions. Light training image.
process TRAIN_SURROGATE {
    tag "$meta.id"
    label 'process_low'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.0"

    input:
    tuple val(meta), path(dataset)

    output:
    tuple val(meta), path("${prefix}"), emit: results
    path 'versions.yml',                emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    prefix   = task.ext.prefix ?: "${meta.id}"
    """
    surrogate-mgem train \\
        --data-dir ${dataset} \\
        --community-id "${meta.community_id}" \\
        --out ${prefix} \\
        --hidden ${meta.hidden} \\
        --n-models ${meta.n_models} \\
        --n-train ${meta.n_train} \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        surrogate-mgem: \$(python -c "import surrogate_mgem; print(surrogate_mgem.__version__)")
        torch: \$(python -c "import torch; print(torch.__version__)")
    END_VERSIONS
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    mkdir -p ${prefix}
    printf '{"community_id": "%s", "mode": "static", "hidden": [64, 64], "n_models": %s, "n_train": %s, "r2_overall": 0.5, "mae_overall": 0.1, "epochs_run_mean": 120.0, "epochs_run_max": 140, "stopped_early_frac": 1.0, "best_val_mean": 0.05, "final_lr_mean": 0.0005, "final_batch_size_max": 128}' "${meta.community_id}" "${meta.n_models}" "${meta.n_train}" > ${prefix}/train_metrics.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
