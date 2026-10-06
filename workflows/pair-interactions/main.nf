// Pairwise metabolic interactions in a community (design spec §13.5, M13; §13.11).
//
// SURVEY draws media over the union of ALL members' active subspaces and solves every
// member at each one, so one set of shards answers the question for every ordered pair
// (G solves a medium, not 2 per pair). RANK_PAIRS turns that into the edge tables and a
// ranking; INTERACTIONS then runs `cfs interactions` per (pair, arm, objective, seed)
// for the top --max_pairs only, because a search costs ~10 core-h a pair and the survey
// costs nothing extra per pair. CROSSEVAL is sharded per pair -- it caches a cobra model
// per genome, and holding a whole community's OOM-kills the task (CLAUDE.md). EDGES
// folds the designed-medium evidence back in.
//
// Reports are rendered locally from the outdir:
//   task pair:report      RUN_DIR=<outdir>   # one pair, in full
//   task community:report RUN_DIR=<outdir>   # every pair, as a network
// See README.md.

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
    for g in ${meta.pair.tokenize(',').join(' ')}; do echo "\$g,\$PWD/gems/\$g.xml" >> roster.csv; done
    ${meta.ceq ? "echo '{\"default\": ${meta.ceq}}' > ceq.json" : ''}
    echo '${groovy.json.JsonOutput.toJson(meta)}' > ${id}/run.json
    # exit 1 = V5 not passed: an outcome recorded in interactions.json, not a failure
    ${params.cfs} interactions --roster roster.csv --labels labels --value value \\
        --behaviour behaviour --communities '${meta.pair}' --seed ${meta.seed} \\
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
    // flat in the work dir (RANK_PAIRS/EDGES stage every shard into one directory),
    // survey/<arm>/shard_<k>.npz where it is published
    publishDir "${params.outdir}/survey", mode: 'copy',
               saveAs: { f -> f.replaceFirst('__', '/') }

    input:
    tuple val(name), val(ceq), val(shard), val(members), path(labels, stageAs: 'labels'), path(value, stageAs: 'value'), path(gems, stageAs: 'gems')
    path script

    output:
    path "${name}__shard_${shard}.npz"

    script:
    def ceq_arg = ceq ? "--ceq ${ceq}" : ''
    """
    ${params.python} ${script} --labels labels --gems gems --value value --pair '${members}' \
        ${ceq_arg} --seed ${params.survey_seed} --shard ${shard} --n ${params.survey_media} \
        --out ${name}__shard_${shard}.npz
    """

    stub:
    "touch ${name}__shard_${shard}.npz"
}

process PRUNE {
    tag "${run.name}"
    label 'process_prune'
    publishDir "${params.outdir}/prune", mode: 'copy'

    input:
    path run
    path gems, stageAs: 'gems'
    path script

    output:
    path "${run.name}.json"

    script:
    "${params.python} ${script} ${run} --gems gems --prune --out ."

    stub:
    "touch ${run.name}.json"
}

process CROSSEVAL {
    tag "${pair}"
    label 'process_crosseval'
    publishDir "${params.outdir}/crosseval", mode: 'copy'

    input:
    tuple val(pair), path(runs, stageAs: 'runs/*')
    path gems, stageAs: 'gems'
    val ceqs
    path script  // an input, so editing crosseval.py invalidates -resume's cache

    output:
    path "${pair}", emit: tables
    // pair-tagged copy: EDGES stages every pair's into one directory, and
    // `robust.csv` from two pairs collides there
    path "${pair}__robust.csv", emit: robust

    script:
    """
    ${params.python} ${script} runs/* --gems gems --ceq '${ceqs}' --out '${pair}'
    cp '${pair}'/robust.csv '${pair}__robust.csv'
    """

    stub:
    "mkdir '${pair}' && touch '${pair}'/media.csv '${pair}'/robust.csv '${pair}__robust.csv"
}

// The edge tables: `edge_summary.py` over the survey, and again with the designed
// media folded in. Two processes rather than one called twice, because the ranking
// has to be available before any search starts.
process RANK_PAIRS {
    label 'process_edges'
    publishDir "${params.outdir}/survey_edges", mode: 'copy'

    input:
    path survey, stageAs: 'survey/*'
    path script

    output:
    path '*.csv', emit: tables

    script:
    "${params.python} ${script} --survey survey --out ."

    stub:
    "printf 'rank,member_a,member_b,pair\\n1,A,B,\"A,B\"\\n' > pairs.csv"
}

process EDGES {
    label 'process_edges'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path survey, stageAs: 'survey/*'
    path robust, stageAs: 'robust/*'
    path script

    output:
    path '*.csv'

    script:
    "${params.python} ${script} --survey survey --robust robust/* --out ."

    stub:
    "touch edges.csv graph_edges.csv relationships.csv pairs.csv"
}

workflow {
    if (params.containsKey('pair')) {
        error "`pair` is now `members` (the whole community; pairs are derived from it). " +
              "Set `members = '${params.pair}'` and `max_pairs` in your config."
    }
    def data = file(params.data, checkIfExists: true)
    def members = params.members.toString().tokenize(',')*.trim()
    if (members.size() < 2) error "--members needs at least two genome ids, got '${params.members}'"
    gems = file("${params.data}/gems", checkIfExists: true)
    log.info "[MEMBERS] ${members.size()}: ${members.join(', ')} " +
             "(${members.size() * (members.size() - 1) / 2 as int} pairs, searching the top ${params.max_pairs})"

    // One survey for the whole community: media over the union of every member's
    // active subspace, every member solved at each one. Shards are seeded by index,
    // so raising --survey_shards with -resume only adds media.
    shards = Channel.fromList(params.arms)
        .combine(Channel.of(0..<(params.survey_shards as int)).flatten())
        .map { arm, k -> [arm.name, arm.ceq ?: null, k, members.join(','),
                          data.resolve(arm.labels), data.resolve(arm.value), gems] }
    SURVEY(shards, file("${projectDir}/survey.py"))

    // Rank the pairs from the survey alone, then spend searches on the top ones.
    RANK_PAIRS(SURVEY.out.collect(), file("${projectDir}/edge_summary.py"))
    pairs = RANK_PAIRS.out.tables.flatten()
        .filter { it.name == 'pairs.csv' }
        .splitCsv(header: true)
        .filter { (it.rank as int) <= (params.max_pairs as int) }
        .map { [it.member_a, it.member_b] }

    // pair x arm x objective x seed; objectives marked inhibited_only skip arms
    // without a c^eq
    jobs = pairs
        .combine(Channel.fromList(params.arms))
        .combine(Channel.fromList(params.objectives))
        .combine(Channel.fromList(params.seeds.toString().tokenize(',')))
        .filter { a, b, arm, obj, seed -> arm.ceq || !obj.inhibited_only }
        .map { a, b, arm, obj, seed ->
            def pair = "${a},${b}"
            def id = "${a}__${b}__${arm.name}__${obj.name}__s${seed}"
            def meta = [pair: pair, arm: arm.name, objective: obj.name, seed: seed as int,
                        ceq: arm.ceq ?: null]
            def args = obj.args + (arm.ceq && obj.inhibited_args ? " ${obj.inhibited_args}" : '')
            [id, meta, args, data.resolve(arm.value), data.resolve(arm.behaviour),
             data.resolve(arm.labels), gems]
        }

    INTERACTIONS(jobs)
    PRUNE(INTERACTIONS.out.run, gems, file("${projectDir}/crosseval.py"))

    // One CROSSEVAL per pair: it caches a cobra model per genome, so one task over a
    // whole community's runs is the documented OOM (CLAUDE.md, §13.5 engineering notes).
    ceqs = params.arms.findAll { it.ceq }.collect { it.ceq }.unique().join(',')
    by_pair = INTERACTIONS.out.run
        .map { run -> [run.name.split('__')[0..1].join('__'), run] }
        .groupTuple()
    CROSSEVAL(by_pair, gems, ceqs, file("${projectDir}/crosseval.py"))

    EDGES(SURVEY.out.collect(), CROSSEVAL.out.robust.collect(),
          file("${projectDir}/edge_summary.py"))
}
