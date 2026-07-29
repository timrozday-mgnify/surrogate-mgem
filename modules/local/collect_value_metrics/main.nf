// Gather every sweep cell's diagnostics.json into one sweep_leaderboard.csv.
// The cell dir name IS the cell_id (unique per samplesheet row), so nothing is
// staged-as-renamed here. Every field is read with .get(): `baseline-rf` never
// writes concavity/Hessian numbers, and checkpoints written before the split-fix
// have no n_val_media/n_train_media/rounds_present -- two runs are only comparable
// when those match, which is why they are columns.
process COLLECT_VALUE_METRICS {
    tag "$meta.id"
    label 'process_single'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.3"

    input:
    tuple val(meta), path(cells)

    output:
    tuple val(meta), path('sweep_leaderboard.csv'), emit: leaderboard
    path 'versions.yml',                            emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    python - <<'PY'
    import glob, json, statistics
    import pandas as pd

    rows = []
    for f in sorted(glob.glob('*/diagnostics.json')):
        d = json.load(open(f))
        per = d.get('per_organism', {})
        cos = {g: v['grad_cosine'] for g, v in per.items()}
        r2 = [v['value_r2'] for v in per.values()]
        conds = [v['hessian_cond_median'] for v in per.values()
                 if v.get('hessian_cond_median') is not None]
        rows.append({
            'cell_id': f.split('/')[0],
            'arch': d.get('arch'),
            'worst_grad_cosine': d.get('worst_grad_cosine'),
            'mean_grad_cosine': statistics.fmean(cos.values()) if cos else None,
            'min_value_r2': min(r2) if r2 else None,
            'median_value_r2': statistics.median(r2) if r2 else None,
            'worst_organism': min(cos, key=cos.get) if cos else None,
            'hessian_cond_median': statistics.median(conds) if conds else None,
            'passed': d.get('passed'),
            'n_val_media': d.get('n_val_media'),
            'n_train_media': d.get('n_train_media'),
            'rounds_present': ','.join(str(r) for r in d.get('rounds_present', [])),
            'n_organisms': len(per),
        })
    frame = pd.DataFrame(rows).sort_values('worst_grad_cosine', ascending=False)
    frame.to_csv('sweep_leaderboard.csv', index=False)
    print(f'{len(rows)} cells -> sweep_leaderboard.csv')
    PY

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        pandas: \$(python -c "import pandas; print(pandas.__version__)")
    END_VERSIONS
    """

    stub:
    """
    printf 'cell_id,arch,worst_grad_cosine,mean_grad_cosine,min_value_r2,median_value_r2,worst_organism,hessian_cond_median,passed,n_val_media,n_train_media,rounds_present,n_organisms\\n' > sweep_leaderboard.csv
    for f in */diagnostics.json; do
        [ -e "\$f" ] && printf '%s,icnn,0.5,0.5,0.5,0.5,g0,,False,,,,1\\n' "\$(dirname "\$f")" >> sweep_leaderboard.csv
    done

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
