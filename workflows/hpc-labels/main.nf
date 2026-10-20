// Label roots + the four heads `workflows/pair-interactions` needs, for a new GEM
// roster (design spec §4.5, §8.6c, §8.6d, §13.11 stage 4').
//
// Two arms, one per ground-truth model: `fba` (plain FBA) and `ceq1mM` (§13.11
// product inhibition at c^eq = 1 mM; 0.1 mM starves 13 of 21 organisms). A c^eq makes
// a different *label root*, not a flag, so each arm gets its own root and its own
// pair of heads. The output names match pair-interactions/stage_data.sh exactly.
//
//   GENERATE(arm, organism)      base 4000 media + the round-1 community pass
//   COMMUNITIES                  the two member lists, drawn once off the roster
//   HOLDOUT                      level-1 trial media; both arms share one set
//   TRAIN_*_INTERIM              the fba heads round 2 needs to walk trajectories
//   fba only, §8.6d round 2:
//     COMMUNITY(community, seed) true-LP n=15 trajectories  <- the dominant cost
//     POOL -> GENERATE_MEDIA(organism)
//   ASSEMBLE(arm)                the published root: base + round 1 (+ round 2)
//   HEAD_A / HEAD_B
//
// Both heads are rebuilt from the round-2 root together: `x_scale` moves with a round
// (P14), so a reused head and a new root disagree about the input coordinate.
//
// Everything runs in the repo venv (../pair-interactions/setup.sh). One organism per
// task and one solver/XLA thread per task: parallelism is the fan-out (CLAUDE.md).

include { GENERATE; GENERATE_MEDIA; ASSEMBLE; COMMUNITIES; HOLDOUT; COMMUNITY; POOL } from './modules.nf'
include { HEAD_A as HEAD_A_INTERIM; HEAD_A } from './modules.nf'
include { HEAD_B as HEAD_B_INTERIM; HEAD_B } from './modules.nf'

workflow {
    def index  = file("${params.data}/metabolite_index.json", checkIfExists: true)
    def roster = file("${params.data}/roster.csv", checkIfExists: true)
    def gems   = file("${params.data}/gems", checkIfExists: true)
    def wanted = params.organisms ? params.organisms.toString().tokenize(',')*.trim() : null
    def byName = params.arms.collectEntries { [it.name, it] }
    def fba    = params.arms.find { !it.ceq }
    if (!fba) error "one arm must be plain FBA (ceq: null): the holdout and round 2 use it"

    // `.toList()` makes these value channels, which is what lets the same label root
    // and the same holdout feed five processes; a queue channel is consumed once.
    organisms = Channel.fromPath(roster).splitCsv(header: true)
        .map { it.genome_id }
        .filter { wanted == null || wanted.contains(it) }
        .toList()
    organisms.subscribe { log.info "[ORGANISMS] ${it.size()} x ${params.arms.size()} arms" }

    // base + round 1, every arm x every organism
    GENERATE(Channel.fromList(params.arms).combine(organisms.flatMap { it })
        .map { arm, gid -> [arm, gid, gems, index] })

    // `labels/<gid>` with the sidecars beside it is already the layout `cfs train-*`
    // reads, so the interim heads and the holdout stage this channel directly -- only
    // the published root has to be assembled.
    // A Map, not a list of pairs: `combine` splices a List item into the tuple it
    // emits, so a channel whose item is a list of files cannot be combined with
    // anything without flattening into one argument per file.
    roots = GENERATE.out.shards
        .map { name, dir, side -> [name, [dir] + side] }
        .groupTuple()
        .map { name, files -> [name, files.flatten()] }
        .toList()
        .map { arms -> arms.collectEntries() }

    COMMUNITIES(roster, file("${projectDir}/make_communities.py"))
    HOLDOUT(roots.map { it[fba.name] }, roster, index, COMMUNITIES.out.bench, gems)
    hold = HOLDOUT.out.npz.first()

    // Interim fba heads. Their only job is to walk the n=15 trajectories whose states
    // round 2 labels; the heads the bundle takes are the r2 ones.
    HEAD_A_INTERIM(roots.combine(hold)
        .map { r, npz -> [fba.value_interim, r[fba.name], index, npz] })
    HEAD_B_INTERIM(roots.map { r -> [fba.behaviour_interim, r[fba.name], index] })

    // One task per (community, seed). `cfs community` names its trajectories after the
    // members, so a per-community fan-out needs no extra disambiguation.
    n15 = COMMUNITIES.out.n15
        .map { it.text.trim().tokenize(';') }
        .flatMap { lists -> lists.withIndex().collect { m, i -> [i, m.trim()] } }
    COMMUNITY(n15
        .combine(Channel.fromList(params.n15_seeds.toString().tokenize(',')))
        .combine(roots)
        .combine(HEAD_A_INTERIM.out.head.map { it[1] }.first())
        .combine(HEAD_B_INTERIM.out.head.map { it[1] }.first())
        .map { idx, members, seed, r, v, b -> [idx, members, seed, r[fba.name], v, b, roster, gems] })

    POOL(COMMUNITY.out.collect(), hold, file("${projectDir}/make_label_pool.py"))
    GENERATE_MEDIA(organisms.flatMap { it }.combine(POOL.out)
        .map { gid, pool -> [gid, gems, index, pool] })

    // Final roots: each arm's base + round 1, plus round 2 for the fba arm. `--media`
    // writes no subspace sidecars, so only the base pass contributes them.
    split = roots.flatMap { r -> r.collect { k, v -> [k, v] } }
        .branch { with_r2: byName[it[0]].round2
                  plain:   true }
    ASSEMBLE(split.with_r2.combine(GENERATE_MEDIA.out.collect().map { [it] })
        .map { name, files, extra -> [byName[name].labels, files, extra] }
        .mix(split.plain.map { name, files -> [byName[name].labels, files, []] }))

    heads = ASSEMBLE.out.labels.map { root, files ->
        [params.arms.find { it.labels == root }, files] }
    HEAD_A(heads.combine(hold).map { arm, files, npz -> [arm.value, files, index, npz] })
    HEAD_B(heads.map { arm, files -> [arm.behaviour, files, index] })
}
