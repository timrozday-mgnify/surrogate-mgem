// Active-learning supplementation for one community. Folds `params.active_rounds`
// discrete `active-round` calls one after another inside a single task (mirroring
// the reference pipeline's in-process iterative merge): each round trains an
// acquisition ensemble, solves a diverse high-uncertainty batch with the real
// MICOM oracle, and appends the new samples to a growing single-community dataset
// that the next round refits on. Needs the heavy solver image (container-only).
process ACTIVE_LEARN {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-data:0.1.0"

    input:
    tuple val(meta), path(dataset), path(roster)

    output:
    tuple val(meta), path('augmented'), emit: dataset
    path 'versions.yml',                emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    cur=${dataset}
    for r in \$(seq 0 \$(( ${params.active_rounds} - 1 ))); do
        surrogate-mgem active-round \\
            --data-dir \$cur \\
            --community-id "${meta.community_id}" \\
            --roster ${roster} \\
            --out round_\$r \\
            --round \$r \\
            $args
        cur=round_\$r
    done
    cp -rL "\$cur" augmented

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        surrogate-mgem: \$(python -c "import surrogate_mgem; print(surrogate_mgem.__version__)")
        micom: \$(python -c "import micom; print(micom.__version__)")
    END_VERSIONS
    """

    stub:
    """
    mkdir -p augmented
    cp -rL ${dataset}/. augmented/ 2>/dev/null || true

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
