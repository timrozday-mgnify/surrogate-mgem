Not a label set — an empty stand-in so `sweep.csv` resolves and `./run.sh -stub`
works straight out of the repo. The stub processes never look inside it (and
`*.parquet` is gitignored, so a fake shard could not be committed anyway).

For a real run, point the samplesheet's `labels` column at a real label root: the
tree `--stage labels` publishes, i.e. `<id>.exchanges.json` / `<id>.subspace.json`
sidecars beside `genome_id=<id>/eps=<e>/part.parquet`.
