// Surrogate-mGEM training sweep.
//
//   generate (sharded) -> merge -> pick top communities
//     -> active-learning supplementation (per community, N discrete rounds)
//     -> model sweep (community x architecture x ensemble-size x train-size)
//     -> leaderboard
//
// Data-size is swept via the `train --n-train` cap (learning curve on a fixed
// dataset), architecture via `--hidden`, ensemble size via `--n-models`.

include { GENERATE_DATA   } from '../modules/local/generate_data/main'
include { MERGE_DATA      } from '../modules/local/merge_data/main'
include { ACTIVE_LEARN    } from '../modules/local/active_learn/main'
include { TRAIN_SURROGATE } from '../modules/local/train_surrogate/main'
include { COLLECT_METRICS } from '../modules/local/collect_metrics/main'

workflow SURROGATE_TRAINING {
    take:
    ch_roster    // file: roster CSV (genome_id, model_path)

    main:
    ch_versions = Channel.empty()

    // The GEMs named by the roster are separate files. Stage them flat and hand the
    // tasks a rewritten roster whose model_path is the bare filename, so it resolves
    // against the task work dir (read_roster resolves relative to the roster).
    ch_rows = Channel.fromPath(ch_roster).splitCsv(header: true)
    ch_models = ch_rows
        .map { row -> file(row.model_path.startsWith('/') ? row.model_path : "${ch_roster.parent}/${row.model_path}", checkIfExists: true) }
        .collect()
        .map { models -> [ models ] }   // wrap: combine() would otherwise spread the list
    ch_flat_roster = ch_rows
        .map { row -> "${row.genome_id},${file(row.model_path).name}" }
        .collectFile(name: 'roster.csv', seed: 'genome_id,model_path', newLine: true, sort: true)
        .first()

    // 1. Seed data: fan out `generate` over HPC shards (deterministic in seed, so
    //    each shard derives the same community list and solves its own slice).
    // `as int`: CLI-supplied params arrive as Strings, and 0..<'2' is not a 2-element range.
    ch_gen = Channel.fromList((0..<(params.num_shards as int)).toList())
        .map { idx -> [ [id: "shard_${idx}", shard: idx] ] }
        .combine(ch_flat_roster)
        .combine(ch_models)
    GENERATE_DATA(ch_gen)
    ch_versions = ch_versions.mix(GENERATE_DATA.out.versions.first())

    // 2. Merge shards into one seed dataset (value channel: reused below).
    ch_merge_in = GENERATE_DATA.out.shard
        .map { meta, d -> d }
        .collect()
        .map { dirs -> [ [id: 'merged'], dirs ] }
    MERGE_DATA(ch_merge_in)
    ch_versions = ch_versions.mix(MERGE_DATA.out.versions)
    ch_dataset  = MERGE_DATA.out.dataset.first()

    // 3. Pick the top communities by feasible-sample count (channel algebra on the
    //    merged samples.csv, mirroring the reference's in-workflow reference pick).
    ch_top_comm = ch_dataset
        .map { meta, d -> file("${d}/samples.csv") }
        .splitCsv(header: true)
        .filter { row -> row.feasible?.toString()?.toLowerCase() in ['true', '1'] }
        .map { row -> row.community_id }
        .collect()
        .map { ids -> ids.countBy { it }.sort { -it.value }.keySet().take(params.n_communities_augment as int) as List }
        .flatten()

    ch_comm_dataset = ch_top_comm
        .combine(ch_dataset)
        .map { cid, meta, d -> [ [id: cid.replaceAll('[^A-Za-z0-9]', '_'), community_id: cid], d ] }

    // 4. Active-learning supplementation (skipped when active_rounds == 0, in which
    //    case the sweep trains on the seed data, filtered per community at train time).
    if ((params.active_rounds as int) > 0) {
        ACTIVE_LEARN(ch_comm_dataset.combine(ch_flat_roster).combine(ch_models).map { meta, d, roster, models -> [ meta, d, roster, models ] })
        ch_versions  = ch_versions.mix(ACTIVE_LEARN.out.versions.first())
        ch_augmented = ACTIVE_LEARN.out.dataset
    } else {
        ch_augmented = ch_comm_dataset
    }

    // 5. Model sweep: community x architecture x ensemble-size x train-size.
    ch_hidden  = Channel.fromList(params.hidden_configs.tokenize(';')*.trim())
    ch_nmodels = Channel.fromList(params.n_models_list.tokenize(',')*.trim())
    ch_sizes   = Channel.fromList(params.train_sizes.tokenize(',')*.trim())

    ch_train_in = ch_augmented
        .combine(ch_hidden)
        .combine(ch_nmodels)
        .combine(ch_sizes)
        .map { meta, d, hid, nm, sz ->
            def cell = "${meta.id}__h${hid.replaceAll(',', 'x')}__k${nm}__n${sz}"
            [ meta + [id: cell, hidden: hid, n_models: nm, n_train: sz], d ]
        }
    TRAIN_SURROGATE(ch_train_in)
    ch_versions = ch_versions.mix(TRAIN_SURROGATE.out.versions.first())

    // 6. Leaderboard over every sweep cell.
    ch_collect_in = TRAIN_SURROGATE.out.results
        .map { meta, d -> d }
        .collect()
        .map { dirs -> [ [id: 'leaderboard'], dirs ] }
    COLLECT_METRICS(ch_collect_in)
    ch_versions = ch_versions.mix(COLLECT_METRICS.out.versions)

    emit:
    leaderboard = COLLECT_METRICS.out.leaderboard
    versions    = ch_versions
}
