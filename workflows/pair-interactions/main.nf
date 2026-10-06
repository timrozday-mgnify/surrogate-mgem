// Metabolic interactions between one pair of genomes (design spec §13.5, M13; §13.11).
// One `cfs interactions` task per (arm, objective, seed) -> CROSSEVAL re-scores every
// designed medium under every model with the true LP. The report is rendered locally
// from the outdir: `task pair:report RUN_DIR=<outdir>`. See README.md.

process INTERACTIONS {
    tag "${id}"
    label 'process_interactions'
    publishDir "${params.outdir}/runs", mode: 'copy'

    input:
    tuple val(id), val(meta), val(args), path(value, stageAs: 'value'), path(behaviour, stageAs: 'behaviour'), path(labels, stageAs: 'labels'), path(gems, stageAs: 'gems')

    output:
    path "${id}", emit: run

    script:
    def ceq = meta.ceq ? "--inhibition ceq.json" : ''
    """
    mkdir ${id}
    echo genome_id,model_path > roster.csv
    for g in ${params.pair.tokenize(',').join(' ')}; do echo "\$g,\$PWD/gems/\$g.xml" >> roster.csv; done
    ${meta.ceq ? "echo '{\"default\": ${meta.ceq}}' > ceq.json" : ''}
    echo '${groovy.json.JsonOutput.toJson(meta)}' > ${id}/run.json
    # exit 1 = V5 not passed: an outcome recorded in interactions.json, not a failure
    ${params.cfs} interactions --roster roster.csv --labels labels --value value \\
        --behaviour behaviour --communities '${params.pair}' --seed ${meta.seed} \\
        ${ceq} ${args} ${params.args} --out ${id} > ${id}/interactions.log 2>&1 \\
        || [ \$? -eq 1 ]
    test -s ${id}/interactions.json || { tail -20 ${id}/interactions.log >&2; exit 1; }
    """

    stub:
    "mkdir ${id} && touch ${id}/interactions.json"
}

process SURVEY {
    tag "${name}#${shard}"
    label 'process_survey'
    publishDir "${params.outdir}/survey", mode: 'copy'

    input:
    tuple val(name), val(ceq), val(shard), path(labels, stageAs: 'labels'), path(value, stageAs: 'value'), path(gems, stageAs: 'gems')
    path script

    output:
    path "${name}/shard_${shard}.npz"

    script:
    def ceq_arg = ceq ? "--ceq ${ceq}" : ''
    """
    mkdir ${name}
    ${params.python} ${script} --labels labels --gems gems --value value --pair '${params.pair}' \
        ${ceq_arg} --seed ${params.survey_seed} --shard ${shard} --n ${params.survey_media} \
        --out ${name}/shard_${shard}.npz
    """

    stub:
    "mkdir ${name} && touch ${name}/shard_${shard}.npz"
}

process CROSSEVAL {
    label 'process_crosseval'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path runs, stageAs: 'runs/*'
    path gems, stageAs: 'gems'
    val ceqs
    path script  // an input, so editing crosseval.py invalidates -resume's cache

    output:
    path 'crosseval', emit: tables

    script:
    "${params.python} ${script} runs/* --gems gems --ceq '${ceqs}' --out crosseval"

    stub:
    "mkdir crosseval && touch crosseval/media.csv"
}

workflow {
    def data = file(params.data, checkIfExists: true)
    gems = file("${params.data}/gems", checkIfExists: true)

    // arm x objective x seed; objectives marked inhibited_only skip arms without c^eq
    jobs = Channel.fromList(params.arms)
        .combine(Channel.fromList(params.objectives))
        .combine(Channel.fromList(params.seeds.toString().tokenize(',')))
        .filter { arm, obj, seed -> arm.ceq || !obj.inhibited_only }
        .map { arm, obj, seed ->
            def id = "${arm.name}__${obj.name}__s${seed}"
            def meta = [arm: arm.name, objective: obj.name, seed: seed as int, ceq: arm.ceq ?: null]
            def args = obj.args + (arm.ceq && obj.inhibited_args ? " ${obj.inhibited_args}" : '')
            [id, meta, args, data.resolve(arm.value), data.resolve(arm.behaviour), data.resolve(arm.labels), gems]
        }

    // The true-LP survey: media drawn from the search's prior, independent of its
    // seeds. Shards are seeded by index, so raising --survey_shards with -resume
    // only adds media.
    shards = Channel.fromList(params.arms)
        .combine(Channel.of(0..<(params.survey_shards as int)).flatten())
        .map { arm, k -> [arm.name, arm.ceq ?: null, k, data.resolve(arm.labels), data.resolve(arm.value), gems] }
    SURVEY(shards, file("${projectDir}/survey.py"))

    INTERACTIONS(jobs)
    ceqs = params.arms.findAll { it.ceq }.collect { it.ceq }.unique().join(',')
    CROSSEVAL(INTERACTIONS.out.run.collect(), gems, ceqs, file("${projectDir}/crosseval.py"))
}
