# surrogate-mgem

Learned per-genome surrogate metabolic models, composed into a community.

The goal is to replace the per-candidate MICOM LP/QP solve that dominates
community media-optimisation and interaction-sampling runtimes with a fast,
differentiable surrogate. Each genome gets its own learned surrogate; surrogates
are composed with an explicit metabolite exchange-balance coupling so that
adding, dropping, or swapping community members needs **no retraining** — a new
genome is embedded from its own model and slotted in.

Only per-member growth and extracellular (exchange) fluxes are predicted, since
every downstream interaction objective (cross-feeding / MES, growth benefit,
minimal-media) depends on those alone — not on internal fluxes.

## Status

Early scaffold. Build is phased with go/no-go validation gates:

- **Phase 0** — training-data generation from real MICOM community solves.
- **Phase 1** — whole-community surrogate (fixed community) + verify-in-loop.
- **Phase 2** — composable per-genome encoder + composition network.
- **Phase 3** — differentiable media optimisation + active learning.

See the design notes for the full rationale and validation strategy.

## Install

```bash
pip install -e ".[dev]"          # model / train / validate / search + tooling
pip install -e ".[dev,data]"     # also the micom/cobra data-generation stack
```

The `data` extra (micom, cobra) is only needed to *generate* training tables;
the model, training, validation and search code run without it.

## Develop

```bash
pre-commit install
pre-commit run --all-files
pytest                 # unit tests (solver-free)
pytest -m slow         # data-generation tests (need the micom/cobra stack)
```

## Layout

```
src/surrogate_mgem/
  data.py      # Phase 0: sample communities + media, solve MICOM, write tables
  encoder.py   # per-genome encoder (exchange capability -> embedding)
  model.py     # composition network + physical-constraint layers
  train.py     # training loop
  validate.py  # validation suite (accuracy, composition, direction, fidelity)
  infer.py     # inference + verify-shortlist helper
  search.py    # surrogate-driven media search with verify-in-loop
  cli.py       # `surrogate-mgem {generate,train,validate,search}`
```
