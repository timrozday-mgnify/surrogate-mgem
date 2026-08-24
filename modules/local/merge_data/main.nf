// Concatenate the per-shard tidy tables into one seed dataset and take the shared
// exchange_universe.json from the shard that wrote it (shard 0). Uses the light
// training image (pandas only, no solver stack).
process MERGE_DATA {
    tag "$meta.id"
    label 'process_low'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.5"

    input:
    tuple val(meta), path(shards, stageAs: 'shard_*')

    output:
    tuple val(meta), path('merged'), emit: dataset
    path 'versions.yml',             emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def min_growth_frac = params.min_growth_frac ?: 0
    """
    python - <<'PY'
    import glob, json, os, shutil, sys
    import pandas as pd

    shard_dirs = sorted(d for d in glob.glob('shard_*') if os.path.isdir(d))
    os.makedirs('merged', exist_ok=True)
    def read(path):
        # An all-empty table is written headerless (a bare newline), so size > 0
        # is not enough -- pandas raises EmptyDataError on it.
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return None

    for name in ['samples', 'media', 'member_growth', 'membership', 'member_exchange']:
        parts = [p for p in (read(f'{d}/{name}.csv') for d in shard_dirs
                             if os.path.exists(f'{d}/{name}.csv')) if p is not None]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(f'merged/{name}.csv', index=False)
    for d in shard_dirs:
        u = f'{d}/exchange_universe.json'
        if os.path.exists(u):
            shutil.copyfile(u, 'merged/exchange_universe.json')
            break

    # Per-community calibrated bound + essential nutrients: every shard that
    # touched a community writes an identical entry, so a union is enough.
    spec = {}
    for d in shard_dirs:
        s = f'{d}/medium_spec.json'
        if os.path.exists(s):
            with open(s) as fh:
                spec.update(json.load(fh))
    if spec:
        with open('merged/medium_spec.json', 'w') as fh:
            json.dump(spec, fh, indent=2)

    # Growth is the training target: if it never varies there is nothing to learn,
    # and every downstream task (active learning, the whole sweep) would burn hours
    # fitting a constant. Fail here instead.
    growth = read('merged/member_growth.csv')
    stats = {'n': 0, 'frac_positive': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    if growth is not None and len(growth):
        g = growth['growth']
        stats = {'n': int(len(g)), 'frac_positive': float((g > 0).mean()),
                 'std': float(g.std()), 'min': float(g.min()), 'max': float(g.max())}
    with open('merged/growth_stats.json', 'w') as fh:
        json.dump(stats, fh, indent=2)
    print('growth stats:', stats)
    if stats['frac_positive'] < ${min_growth_frac}:
        sys.exit(
            f"Only {100 * stats['frac_positive']:.2f}% of samples grew "
            f"(need >= {100 * ${min_growth_frac}:.2f}%). The media carry no growth signal: "
            "check --sampler (use 'titrate') and that max_uptake is not far above the "
            "saturation point. Set params.min_growth_frac = 0 to bypass this check."
        )
    PY

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        pandas: \$(python -c "import pandas; print(pandas.__version__)")
    END_VERSIONS
    """

    stub:
    """
    mkdir -p merged
    for t in samples media member_growth membership member_exchange; do
        first=\$(ls shard_*/\$t.csv 2>/dev/null | head -1)
        if [ -n "\$first" ]; then cp "\$first" merged/\$t.csv; else touch merged/\$t.csv; fi
    done
    first_u=\$(ls shard_*/exchange_universe.json 2>/dev/null | head -1)
    if [ -n "\$first_u" ]; then cp "\$first_u" merged/exchange_universe.json; fi
    first_s=\$(ls shard_*/medium_spec.json 2>/dev/null | head -1)
    if [ -n "\$first_s" ]; then cp "\$first_s" merged/medium_spec.json; fi
    printf '{"n": 2, "frac_positive": 1.0, "std": 0.05, "min": 0.5, "max": 0.6}' > merged/growth_stats.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
