// M1 (plan §5.2): concatenate the per-organism degeneracy surveys and emit the
// single roster-wide D4 recommendation (clean -> plain FBA / localised -> weak reg
// / genuine -> elastic net). Advisory: the human records the D4 decision. Light
// train image (pandas + cfs, no solver stack).
process COLLECT_D4 {
    tag "$meta.id"
    label 'process_single'

    container "ghcr.io/timrozday-mgnify/surrogate-mgem-train:0.1.3"

    input:
    tuple val(meta), path(surveys, stageAs: 'survey_*.csv')

    output:
    path 'd4_recommendation.json', emit: recommendation
    path 'degeneracy_all.csv',     emit: combined
    path 'versions.yml',           emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    python - <<'PY'
    import glob, json
    import pandas as pd
    from cfs.validate.degeneracy import recommend_d4

    frames = [pd.read_csv(p) for p in sorted(glob.glob('survey_*.csv'))]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv('degeneracy_all.csv', index=False)
    rec = recommend_d4(combined)
    with open('d4_recommendation.json', 'w') as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps(rec, indent=2))
    PY

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        pandas: \$(python -c "import pandas; print(pandas.__version__)")
    END_VERSIONS
    """

    stub:
    """
    printf '{"verdict": "clean", "action": "plain FBA for Head B; no regularisation needed"}' > d4_recommendation.json
    touch degeneracy_all.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: 3.11
    END_VERSIONS
    """
}
