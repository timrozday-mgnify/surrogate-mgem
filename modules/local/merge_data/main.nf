// Concatenate the per-shard tidy tables into one seed dataset and take the shared
// exchange_universe.json from the shard that wrote it (shard 0). Uses the light
// training image (pandas only, no solver stack).
process MERGE_DATA {
    tag "$meta.id"
    label 'process_low'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.0"

    input:
    tuple val(meta), path(shards, stageAs: 'shard_*')

    output:
    tuple val(meta), path('merged'), emit: dataset
    path 'versions.yml',             emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    python - <<'PY'
    import glob, json, os, shutil
    import pandas as pd

    shard_dirs = sorted(d for d in glob.glob('shard_*') if os.path.isdir(d))
    os.makedirs('merged', exist_ok=True)
    for name in ['samples', 'media', 'member_growth', 'membership', 'member_exchange']:
        parts = [pd.read_csv(f'{d}/{name}.csv') for d in shard_dirs
                 if os.path.exists(f'{d}/{name}.csv') and os.path.getsize(f'{d}/{name}.csv')]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(f'merged/{name}.csv', index=False)
    for d in shard_dirs:
        u = f'{d}/exchange_universe.json'
        if os.path.exists(u):
            shutil.copyfile(u, 'merged/exchange_universe.json')
            break
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

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
