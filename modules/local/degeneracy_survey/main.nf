// M1 (plan §5.2): per-organism exchange-FVA degeneracy survey. Fanned out one
// task per genome because FVA over all exchanges is the cost (~thousands of LPs
// per model); 20 organisms run in parallel rather than serially. Each task gets
// one model and writes a one-row roster for it. The roster-wide D4 recommendation
// is assembled downstream by COLLECT_D4. Heavy data image (cobra) -- container-only.
process DEGENERACY_SURVEY {
    tag "$meta.id"
    label 'process_high'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-data:0.1.2"

    input:
    tuple val(meta), path(model)

    output:
    tuple val(meta), path("${meta.id}.degeneracy.csv"), emit: survey
    path 'versions.yml',                                emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    printf 'genome_id,model_path\\n${meta.id},${model}\\n' > roster.csv
    cfs degeneracy --roster roster.csv --outdir survey $args
    cp survey/${meta.id}.degeneracy.csv ${meta.id}.degeneracy.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        cobra: \$(python -c "import cobra; print(cobra.__version__)")
    END_VERSIONS
    """

    stub:
    """
    printf 'genome_id,medium,alpha,exchange,range\\n${meta.id},0,1.0,EX_a_e,0.0\\n' > ${meta.id}.degeneracy.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
