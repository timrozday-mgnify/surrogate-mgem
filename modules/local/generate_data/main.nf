// Generate one HPC shard of surrogate training data. Solves the MICOM communities
// assigned to this shard (ci % num_shards == shard_index); shard 0 also writes the
// shared exchange_universe.json. Needs the heavy micom/cobra solver image — no
// bioconda package, so this module is container-only (conda unsupported).
process GENERATE_DATA {
    tag "$meta.id"
    label 'process_high'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-data:0.1.1"

    input:
    tuple val(meta), path(roster)

    output:
    tuple val(meta), path("shard_${meta.shard}"), emit: shard
    path "versions.yml",                          emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    surrogate-mgem generate \\
        --roster ${roster} \\
        --out shard_${meta.shard} \\
        --num-shards ${params.num_shards} \\
        --shard-index ${meta.shard} \\
        --workers ${task.cpus} \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        surrogate-mgem: \$(python -c "import surrogate_mgem; print(surrogate_mgem.__version__)")
        micom: \$(python -c "import micom; print(micom.__version__)")
    END_VERSIONS
    """

    stub:
    """
    mkdir -p shard_${meta.shard}
    printf 'sample_id,community_id,n_members,feasible,community_growth\\n0,gA+gB,2,True,1.0\\n1,gA+gB,2,True,1.1\\n' > shard_${meta.shard}/samples.csv
    printf 'sample_id,exchange_id,uptake\\n0,EX_a_m,1.0\\n1,EX_a_m,2.0\\n' > shard_${meta.shard}/media.csv
    printf 'sample_id,genome_id,growth\\n0,gA,0.5\\n0,gB,0.5\\n1,gA,0.6\\n1,gB,0.6\\n' > shard_${meta.shard}/member_growth.csv
    printf 'sample_id,genome_id\\n0,gA\\n0,gB\\n1,gA\\n1,gB\\n' > shard_${meta.shard}/membership.csv
    touch shard_${meta.shard}/member_exchange.csv
    if [ ${meta.shard} -eq 0 ]; then printf '{"medium_exchanges": ["EX_a_m"], "member_exchanges": ["EX_a_e"]}' > shard_${meta.shard}/exchange_universe.json; fi

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
