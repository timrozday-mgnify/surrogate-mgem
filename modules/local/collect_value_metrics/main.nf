// Gather every sweep task's diagnostics.json into one sweep_leaderboard.csv, one
// row per CELL. A cell is fanned out one organism per task, so its dirs are named
// `<cell_id>__<genome_id>` (or plain `<cell_id>` whole-stack) and are regrouped
// here by stripping the genome_id the diagnostics themselves name. Every field is read with .get(): `baseline-rf` never
// writes concavity/Hessian numbers, and checkpoints written before the split-fix
// have no n_val_media/n_train_media/rounds_present -- two runs are only comparable
// when those match, which is why they are columns.
process COLLECT_VALUE_METRICS {
    tag "$meta.id"
    label 'process_single'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.5"

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
    from collections import defaultdict

    import pandas as pd

    # The sweep runs one task per (cell, organism), so a cell's diagnostics arrive
    # as several dirs named `<cell_id>__<genome_id>` (one dir, `<cell_id>`, when the
    # cell was trained whole-stack). Merge them back per cell: the media split is
    # by `medium_id` and identical across organisms, so the merged `per_organism`
    # is what one stacked run would have written.
    cells = defaultdict(lambda: {'per': {}, 'meta': {}})
    for f in sorted(glob.glob('*/diagnostics.json')):
        d = json.load(open(f))
        name = f.split('/')[0]
        per = d.get('per_organism', {})
        cell = name
        for g in per:
            cell = cell[: -len(f'__{g}')] if cell.endswith(f'__{g}') else cell
        cells[cell]['per'].update(per)
        cells[cell]['meta'] = d

    rows = []
    for cell, got in cells.items():
        d, per = got['meta'], got['per']
        cos = {g: v['grad_cosine'] for g, v in per.items()}
        r2 = [v['value_r2'] for v in per.values()]
        conds = [v['hessian_cond_median'] for v in per.values()
                 if v.get('hessian_cond_median') is not None]
        worst = min(cos.values()) if cos else None
        rows.append({
            'cell_id': cell,
            'arch': d.get('arch'),
            'worst_grad_cosine': worst,
            'mean_grad_cosine': statistics.fmean(cos.values()) if cos else None,
            'min_value_r2': min(r2) if r2 else None,
            'median_value_r2': statistics.median(r2) if r2 else None,
            'worst_organism': min(cos, key=cos.get) if cos else None,
            'hessian_cond_median': statistics.median(conds) if conds else None,
            # Per-organism tasks each gate their own head, so the cell passes only
            # if every one of them did.
            'passed': bool(worst is not None and worst > 0.99),
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
