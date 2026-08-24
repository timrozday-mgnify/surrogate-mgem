// M0 (plan §2, §3.0): EGC pre-flight + MEMOTE over the whole roster, then freeze
// the shared/private metabolite index. One task over all models (the index is a
// union across the roster, like shard 0 writing exchange_universe.json). Hard
// gate: `cfs qc` exits non-zero if any model generates ATP from nothing.
// Needs the heavy data image (cobra + the memote CLI) -- container-only.
process QC_MODELS {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-data:0.1.5"

    input:
    tuple val(meta), path(roster), path(models)

    output:
    path 'qc_summary.json',       emit: summary
    path 'metabolite_index.json', emit: index
    path '*.memote.html',         emit: memote,  optional: true
    path 'versions.yml',          emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    cfs qc --roster ${roster} --outdir . $args
    cfs freeze-index --roster ${roster} --out metabolite_index.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        surrogate-mgem: \$(python -c "import surrogate_mgem; print(surrogate_mgem.__version__)")
        cobra: \$(python -c "import cobra; print(cobra.__version__)")
    END_VERSIONS
    """

    stub:
    """
    printf '{"gA": {"egc": false, "memote_report": "gA.memote.html"}}' > qc_summary.json
    printf '{"genome_ids": ["gA"], "index": ["EX_a_e"], "shared": [], "private": ["EX_a_e"], "mask": [[1]]}' > metabolite_index.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
