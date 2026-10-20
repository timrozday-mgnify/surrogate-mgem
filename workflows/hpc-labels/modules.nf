// Processes for the label/head chain. Aliased in main.nf, because the two training
// stages (interim fba heads, then the round-2 heads) are the same command run twice
// and Nextflow invokes a process once per workflow.

// base + round 1 in one task: `--round 1` rewrites this organism's subspace sidecars,
// so the two passes must not race for them.
process GENERATE {
    tag "${arm.name}:${gid}"
    label 'process_generate'

    input:
    tuple val(arm), val(gid), path(gems, stageAs: 'gems'), path(index)

    output:
    tuple val(arm.name), path(gid), path("${gid}.*.json"), emit: shards

    script:
    def ceq = arm.ceq ? "--inhibition ceq.json" : ''
    """
    printf 'genome_id,model_path\\n%s,%s\\n' ${gid} \$PWD/gems/${gid}.xml > roster.csv
    ${arm.ceq ? "echo '{\"default\": ${arm.ceq}}' > ceq.json" : ''}
    # --mid-mu 0: labels_p4 predates the B2 mid-mu stratum, which was refuted and is
    # still the default 0.15. No --scales: the §4.7 demand probe anchors a new GEM.
    ${params.cfs} generate --roster roster.csv --index ${index} --outdir . \\
        --n-media ${params.n_media} --mid-mu 0 ${ceq} --seed 0
    ${params.cfs} generate --roster roster.csv --index ${index} --outdir . \\
        --n-media ${params.round1_media} --mid-mu 0 ${ceq} \\
        --round 1 --bg-perturb ${params.round1_bg} --seed 1
    # one per organism and all identical; tagged so 21 can share one directory
    if [ -f inhibition.json ]; then mv inhibition.json ${gid}.inhibition.json; fi
    test \$(ls ${gid}/*/part.parquet | wc -l) -eq 3
    test \$(ls ${gid}/*/part.round1.parquet | wc -l) -eq 3
    """

    stub:
    """
    mkdir -p ${gid}/eps_0.001 && touch ${gid}/eps_0.001/part.parquet
    touch ${gid}.subspace.json ${gid}.exchanges.json
    """
}

// §8.6d round 2: label exactly the media the composition visits. No design, so no
// subspace sidecars -- the parquet is the only output. Prefixed, because the base
// run's `<gid>` is merged with this one by ASSEMBLE, not by a staging directory.
process GENERATE_MEDIA {
    tag "${gid}"
    label 'process_generate_media'

    input:
    tuple val(gid), path(gems, stageAs: 'gems'), path(index), path(pool)

    output:
    path "round2/${gid}"

    script:
    """
    mkdir -p round2
    printf 'genome_id,model_path\\n%s,%s\\n' ${gid} \$PWD/gems/${gid}.xml > roster.csv
    ${params.cfs} generate --roster roster.csv --index ${index} --outdir round2 \\
        --media ${pool} --round 2 --seed 2
    test \$(ls round2/${gid}/*/part.round2.parquet | wc -l) -eq 3
    """

    stub:
    "mkdir -p round2/${gid}/eps_0.001 && touch round2/${gid}/eps_0.001/part.round2.parquet"
}

// The deliverable root, in the layout `cfs train-*` and stage_data.sh read: one
// directory per organism, the sidecars beside them, one index_hash throughout (P13).
process ASSEMBLE {
    tag "${root}"
    label 'process_single'
    publishDir "${params.outdir}", mode: 'link'

    input:
    tuple val(root), path(shards, stageAs: 'base/*'), path(extra, stageAs: 'round/*')

    output:
    tuple val(root), path("${root}/*"), emit: labels

    script:
    """
    mkdir ${root}
    cp -RL base/* ${root}/
    # round N's parquet goes beside the base run's, inside the same eps directory
    for d in round/*; do
        if [ -d "\$d" ]; then cp -RL "\$d"/* ${root}/\$(basename "\$d")/; fi
    done
    # the 21 tagged copies are identical, so moving each onto the canonical name
    # both installs it and clears the rest
    for f in ${root}/*.inhibition.json; do
        if [ -e "\$f" ]; then mv "\$f" ${root}/inhibition.json; fi
    done
    ls ${root} | wc -l
    """

    stub:
    "mkdir -p ${root} && touch ${root}/placeholder"
}

process COMMUNITIES {
    label 'process_single'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path roster
    path script

    output:
    path 'communities.txt',   emit: bench
    path 'communities15.txt', emit: n15

    script:
    """
    ${params.python} ${script} ${roster} ${params.community_seed} \\
        communities.txt communities15.txt \\
        --n15-sets ${params.n15_sets} --n15-size ${params.n15_size}
    """

    stub:
    "printf 'A,B' > communities.txt; printf 'A,B' > communities15.txt"
}

// Level-1 trial media (§8.6c). Both arms select cuts against the same point set:
// `_trial_points` builds a theta half from the medium, so one npz serves either model.
process HOLDOUT {
    label 'process_holdout'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path labels, stageAs: 'labels/*'
    path roster
    path index
    path communities
    path gems, stageAs: 'gems'

    output:
    path 'holdout_community', emit: dir
    path 'holdout_community/community_holdout.npz', emit: npz

    script:
    """
    sed 's#,.*/\\([^/]*\\)\\.xml#,'"\$PWD"'/gems/\\1.xml#' ${roster} > local_roster.csv
    ${params.cfs} community-holdout make --roster local_roster.csv --labels labels \\
        --index ${index} --communities "\$(cat ${communities})" \\
        --n-media ${params.holdout_media} --seed ${params.holdout_seed} --out holdout_community
    """

    stub:
    "mkdir holdout_community && touch holdout_community/community_holdout.npz"
}

// HEAD_A/HEAD_B rather than TRAIN_VALUE/TRAIN_BEHAVIOUR: the repo-root
// conf/modules.config has a `withName: 'TRAIN_VALUE|BASELINE_RF'` rule for the
// legacy sweep, and the launch-directory config wins -- under that name the heads
// are published into `<outdir>/sweep/` instead.
//
// Head A: frozen level-1 label tangents + the intercept repair, shipped colder than
// it selected cuts at (§8.6c). `--epochs 0` is a supported mode; exit 1 is the M3
// gate being unmet, which is an outcome in diagnostics.json and not a failure.
process HEAD_A {
    tag "${name}"
    label 'process_head_a'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    tuple val(name), path(labels, stageAs: 'labels/*'), path(index), path(holdout)

    output:
    tuple val(name), path(name), emit: head

    script:
    """
    ${params.cfs} train-value --labels labels --index ${index} --out ${name} \\
        --arch groupmax-u --width 1 --depth 1 --epochs 0 --lr 3e-3 --w-grad 10 \\
        --gm-group 1000 --gm-temp 0.01 --gm-eval-temp 0.0001 --gm-init labels \\
        --gm-select level1 --gm-trial-media ${holdout} --gm-repair --seed 0 \\
        ${params.value_args} > train.log 2>&1 || [ \$? -eq 1 ]
    mkdir -p ${name} && cp train.log ${name}/train.log
    test -s ${name}/diagnostics.json || { tail -30 train.log >&2; exit 1; }
    """

    stub:
    "mkdir ${name} && touch ${name}/diagnostics.json"
}

process HEAD_B {
    tag "${name}"
    label 'process_head_b'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    tuple val(name), path(labels, stageAs: 'labels/*'), path(index)

    output:
    tuple val(name), path(name), emit: head

    script:
    """
    ${params.cfs} train-behaviour --labels labels --index ${index} --out ${name} \\
        --width 256 --depth 3 --epochs ${params.behaviour_epochs} --lr 1e-3 --seed 0 \\
        > train.log 2>&1
    mkdir -p ${name} && cp train.log ${name}/train.log
    # Head B's is `behaviour_diagnostics.json`; Head A's is `diagnostics.json`
    test -s ${name}/behaviour_diagnostics.json || { tail -30 train.log >&2; exit 1; }
    """

    stub:
    "mkdir ${name} && touch ${name}/behaviour_diagnostics.json"
}

// One task per (n=15 community, seed): the true LP is solved for every member at
// every step, so this is the run's dominant cost and the one worth fanning out.
process COMMUNITY {
    tag "n15_${idx}_s${seed}"
    label 'process_community'

    input:
    tuple val(idx), val(members), val(seed), path(labels, stageAs: 'labels/*'), path(value), path(behaviour), path(roster), path(gems, stageAs: 'gems')

    output:
    path "n15_${idx}_s${seed}"

    script:
    """
    sed 's#,.*/\\([^/]*\\)\\.xml#,'"\$PWD"'/gems/\\1.xml#' ${roster} > local_roster.csv
    ${params.cfs} community --roster local_roster.csv --labels labels \\
        --value ${value} --behaviour ${behaviour} --communities '${members}' \\
        --seed ${seed} --out n15_${idx}_s${seed} > n15_${idx}_s${seed}.log 2>&1
    test -n "\$(ls n15_${idx}_s${seed}/trajectory_*.npz)"
    """

    stub:
    "mkdir n15_${idx}_s${seed} && touch n15_${idx}_s${seed}/trajectory_0_x.npz"
}

process POOL {
    label 'process_single'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path runs, stageAs: 'runs/*'
    path holdout
    path script

    output:
    path 'label_pool_n15.npz'

    script:
    "${params.python} ${script} runs/* --holdout ${holdout} --stride ${params.pool_stride} --out label_pool_n15.npz"

    stub:
    "touch label_pool_n15.npz"
}
