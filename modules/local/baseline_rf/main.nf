// M3b/§7.4: the random-forest floor/ceiling on the same split and gate as a
// Head A cell. It is what says whether a poor ICNN score is an architecture deficit
// or a label ceiling -- R^2 0.979 against the ICNN's 0.538 is why the sweep keeps it
// in. `arch = rf` rows in the samplesheet route here; no `--arch` flag exists.
// Its gradients are finite differences, so `--delta` matters (the worst-organism
// cosine is U-shaped in it); the row carries that in `ext.args`.
process BASELINE_RF {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.5"

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
    // No `|| true` here: `cfs baseline-rf` is a measurement and always exits 0,
    // however it scores, so a non-zero status is a genuine fault.
    """
    cfs baseline-rf \\
        --labels ${labels} \\
        --index ${index} \\
        --out ${prefix} \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        scikit-learn: \$(python -c "import sklearn; print(sklearn.__version__)")
    END_VERSIONS
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    mkdir -p ${prefix}
    printf '{"worst_grad_cosine": 0.4, "passed": false, "arch": "random-forest", "per_organism": {"${meta.organism ?: 'g0'}": {"grad_cosine": 0.4, "value_r2": 0.97}}}' > ${prefix}/diagnostics.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
