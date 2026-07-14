// Gather every sweep cell's train_metrics.json into one leaderboard.csv
// (community x architecture x ensemble-size x train-size x R2/MAE). Light image.
process COLLECT_METRICS {
    tag "$meta.id"
    label 'process_single'

    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.0-sif' :
        'ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.0' }"

    input:
    tuple val(meta), path(results)

    output:
    tuple val(meta), path('leaderboard.csv'), emit: leaderboard
    path 'versions.yml',                      emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    python - <<'PY'
    import glob, json
    import pandas as pd

    rows = []
    for f in glob.glob('*/train_metrics.json'):
        d = json.load(open(f))
        d.pop('r2_per_member', None)
        d['hidden'] = 'x'.join(str(x) for x in d.get('hidden', []))
        rows.append(d)
    frame = pd.DataFrame(rows)
    if 'r2_overall' in frame:
        frame = frame.sort_values('r2_overall', ascending=False)
    frame.to_csv('leaderboard.csv', index=False)
    print(f'{len(rows)} cells -> leaderboard.csv')
    PY

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        pandas: \$(python -c "import pandas; print(pandas.__version__)")
    END_VERSIONS
    """

    stub:
    """
    printf 'community_id,hidden,n_models,n_train,r2_overall,mae_overall,epochs_run_mean,stopped_early_frac,best_val_mean,final_batch_size_max\\n' > leaderboard.csv
    for f in */train_metrics.json; do
        [ -e "\$f" ] && printf 'stub,64x64,2,40,0.5,0.1,120.0,1.0,0.05,128\\n' >> leaderboard.csv
    done

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
