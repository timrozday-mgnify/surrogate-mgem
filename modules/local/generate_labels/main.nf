// M2/§4.5: bulk ground-truth label generation for one organism -> parquet.
// `cfs generate` builds the active subspace (§4.2), the stratified-Sobol design
// (§4.3-4.4) and solves the elastic-net QP over media x alpha x eps (§5.4),
// sharding to `<id>/eps_<e>/part.parquet`. Fanned out one task per
// genome exactly like DEGENERACY_SURVEY -- each task writes a one-row roster for
// its own model. Single-threaded (HiGHS is pinned to threads=1 for repeatable
// labels), so parallelism is across organisms, not inside one.
// Heavy data image (cobra + highspy + pyarrow) -- container-only.
process GENERATE_LABELS {
    tag "$meta.id"
    label 'process_low'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-data:0.1.5"

    input:
    // `scales` and `focus` are optional -- pass [] to omit. Both are staged files,
    // so the flags are built below rather than in conf/modules.config's ext.args.
    tuple val(meta), path(model), path(index), path(scales), path(focus)

    output:
    // Written straight into the task dir, not a subdir: publishDir keeps each
    // output's relative path, so `--outdir labels` would publish to labels/labels/.
    tuple val(meta), path("${meta.id}/"),          emit: shards
    tuple val(meta), path("${meta.id}.*.json"),    emit: metadata
    path 'versions.yml',                           emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def opt = (scales ? " --scales ${scales}" : '') + (focus ? " --focus-weights ${focus}" : '')
    """
    printf 'genome_id,model_path\\n${meta.id},${model}\\n' > roster.csv
    cfs generate --roster roster.csv --index ${index} --outdir . $args${opt}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        cobra: \$(python -c "import cobra; print(cobra.__version__)")
        highspy: \$(python -c "from importlib.metadata import version; print(version('highspy'))")
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${meta.id}/eps_0.001
    touch ${meta.id}/eps_0.001/part.parquet
    printf '{"index_hash": "stub", "exchanges": ["EX_a_e"]}' > ${meta.id}.exchanges.json
    printf '{"${meta.id}": {"active": ["EX_a_e"], "background": [], "sensitivity": {"EX_a_e": 1.0}, "mu_rich": 1.0}}' > ${meta.id}.subspace.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
