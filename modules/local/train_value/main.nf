// M3b/§7.4: train one Head A cell (`cfs train-value`) on a label set.
// One task per sweep-samplesheet row; the row's flags arrive as `ext.args`, so the
// cell's exact command is greppable in the samplesheet rather than assembled from a
// dozen params. Reads label shards only -- no roster, no GEM, no solver -- so this
// is the light train image plus the `jax` extra.
process TRAIN_VALUE {
    tag "$meta.id"
    label 'process_high'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.7"

    input:
    tuple val(meta), path(labels), path(index)

    output:
    tuple val(meta), path("${prefix}"), emit: results
    path 'versions.yml',                emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    // `_shard_organisms` splits the organism axis over host devices (1.8x on CPU),
    // but only when the device count DIVIDES the organism count -- otherwise it logs
    // and silently runs unsharded. Read once at backend init, hence the env var.
    def xla = (params.xla_devices as int) > 0
        ? "export XLA_FLAGS=--xla_force_host_platform_device_count=${params.xla_devices}"
        : ''
    """
    ${xla}
    # `cfs train-value` exits 1 whenever the 0.99 gate is unmet (cli.py) and the best
    # cell measured to date is 0.733 -- an honest exit code would fail every task in
    # the sweep. A cell that produced no diagnostics is the real failure, so gate on
    # that instead and let the leaderboard report the score.
    #
    # Tolerate ONLY exit 1. A blanket `|| true` also swallowed 137 (OOM kill), which
    # is the one exit status `conf/base.config` retries on with more memory: the
    # deepset cells were dying on the kernel and being reported as "no diagnostics"
    # with the real cause three lines up in .command.err.
    rc=0
    cfs train-value \\
        --labels ${labels} \\
        --index ${index} \\
        --out ${prefix} \\
        --arch ${meta.arch} \\
        $args || rc=\$?
    [ "\$rc" -le 1 ] || exit "\$rc"
    test -s ${prefix}/diagnostics.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        jax: \$(python -c "import jax; print(jax.__version__)")
        equinox: \$(python -c "import equinox; print(equinox.__version__)")
    END_VERSIONS
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    mkdir -p ${prefix}
    printf '{"worst_grad_cosine": 0.5, "passed": false, "arch": "${meta.arch}", "per_organism": {"${meta.organism ?: 'g0'}": {"grad_cosine": 0.5, "value_r2": 0.5}}}' > ${prefix}/diagnostics.json
    touch ${prefix}/value_heads.eqx ${prefix}/value_heads.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
